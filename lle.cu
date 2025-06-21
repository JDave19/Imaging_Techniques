#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <npp.h>

#define img_wid 800
#define img_hei 534
#define img_cha 3
#define iterations 5
#define m 0.6f
#define z 0.9f


// Defining the kernel size
dim3 blockSize(32,32);
dim3 gridSize((img_wid + blockSize.x -1 )/ blockSize.x,
              (img_hei + blockSize.y -1 )/ blockSize.y);

/************************************ CUDA Kernels ***************************************/
__global__ void invert(uchar* in, float* out, int wid, int hei){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < wid && y < hei){
        int idx = y * wid + x;
        out[idx * 3 + 0] = static_cast<float>(in[idx * 3 + 0]);
        out[idx * 3 + 1] = static_cast<float>(in[idx * 3 + 1]);
        out[idx * 3 + 2] = static_cast<float>(in[idx * 3 + 2]);
    }
}
__global__ void dark_cha(float *in, float *out, int wid, int hei){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < wid && y < hei){
        int idx = y * wid + x;
        float r = in[idx * 3 + 0];
        float g = in[idx * 3 + 1];
        float b = in[idx * 3 + 2];
        
        out[idx] = 255.0f - (fminf(r, fminf(g,b)));
    }
}
__global__ void enhance(float *air, float *input_img, unsigned char *enhanced, int wid, int hei) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if ( x < wid && y < hei){
        int idx = y * wid + x;

        float ratio = z / (255.0f * m);
        float factor = 1 + powf((ratio * air[idx]), 2);

        float r = input_img[idx * 3 + 0] * factor;
        float g = input_img[idx * 3 + 1] * factor;
        float b = input_img[idx * 3 + 2] * factor;

        enhanced[idx * 3 + 0] = static_cast<uchar>(min(255.0f, max(0.0f, r)));
        enhanced[idx * 3 + 1] = static_cast<uchar>(min(255.0f, max(0.0f, g)));
        enhanced[idx * 3 + 2] = static_cast<uchar>(min(255.0f, max(0.0f, b)));

    }
}
/************************************************************************/
int main(){
    /////// Vars ///////
    cv::Mat img;

    // uchar vars
    uchar *d_img, *h_out, *d_enhanced;

    // float vars
    float *d_float_img, *d_air, *d_air_2;

    /////// Memory Allocations ///////
    cudaMalloc((void**)&d_img, img_wid * img_hei * img_cha * sizeof(uchar));
    cudaMalloc((void**)&d_float_img, img_wid * img_hei * img_cha * sizeof(float));
    cudaMalloc((void**)&d_air, img_wid * img_hei * sizeof(float));
    cudaMalloc((void**)&d_air_2, img_wid * img_hei * sizeof(float));
    cudaMalloc((void**)&d_enhanced, img_wid * img_hei * img_cha * sizeof(uchar));
    h_out = (uchar *)malloc(img_wid * img_hei * img_cha * sizeof(uchar));

    // Read the image as RGB
    img = cv::imread("low_light_img.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Copying to device
    cudaMemcpy(d_img, img.data, img_wid * img_hei * img_cha * sizeof(uchar), cudaMemcpyHostToDevice);
    invert<<<gridSize, blockSize>>>(d_img, d_float_img, img_wid, img_hei);
    dark_cha<<<gridSize, blockSize>>>(d_float_img, d_air, img_wid, img_hei);
    NppiSize roi = {img_wid, img_hei};
    NppiPoint anchor = {0, 0};

    for (int i = 0; i < iterations; i++) {
        nppiFilterGaussBorder_32f_C1R(
            d_air, img_wid * sizeof(float),
            roi,
            anchor,
            d_air_2, img_wid * sizeof(float),
            roi,
            NPP_MASK_SIZE_3_X_3,
            NPP_BORDER_REPLICATE
        );
        // Swap pointers for next iteration
        std::swap(d_air, d_air_2);
    }
    enhance<<<gridSize, blockSize>>>(d_air, d_float_img, d_enhanced, img_wid, img_hei);
    cudaMemcpy(h_out, d_enhanced, img_wid * img_hei * img_cha * sizeof(uchar), cudaMemcpyDeviceToHost);

    cv:: Mat output(img_hei, img_wid, CV_8UC3, h_out);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    cv::imshow("Enhanced", output);
    cv::imshow("Input", img);
    cv::waitKey();
    cv::destroyAllWindows();

    cudaFree(d_img);
    cudaFree(d_float_img);
    cudaFree(d_air);
    cudaFree(d_enhanced);
    free(h_out);
    return 0;

}
