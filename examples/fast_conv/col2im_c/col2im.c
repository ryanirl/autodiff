#include <stdio.h>
#include <math.h>

void col2im_cpu(double *cols, double *col2img, int N, int C, int H, 
                int W, int FH, int FW, int stride, int padding) {

    // Here H and W are already adjusted for padding.
    int out_H = ((H - FH) / stride) + 1;
    int out_W = ((W - FW) / stride) + 1;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int fh = 0; fh < FH; ++fh) {
                for (int fw = 0; fw < FW; ++fw) {
                    for (int h = 0; h < out_H; ++h) {
                        for (int w = 0; w < out_W; ++w) {
                            // To understand indexing flattened arrays I recommend this stackoverflow post:
                            // https://stackoverflow.com/questions/29142417/4d-position-from-1d-index/29148148
                            
                            int h_itter = stride * h + fh;
                            int w_itter = stride * w + fw;

                            // You could shorten the next two lines and remove some complexity if you want.
                            int col2img_itter = w_itter + (h_itter * W) + (c * W * H) + (n * W * H * C);

                            int cols_itter = w + (h * out_W) + (n * out_W * out_H) + (fw * out_W * out_H * N) +
                                            (fh * out_W * out_H * N * FW) + (c * out_W * out_H * N * FW * FH);

                            double val = cols[cols_itter];

                            col2img[col2img_itter] += val;

                        }
                    }
                }
            }
        }
    }
}





