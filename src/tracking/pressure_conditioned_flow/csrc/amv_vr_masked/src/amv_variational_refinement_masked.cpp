/*
 * Pressure-conditioned AMV masked VariationalRefinement fork.
 * Exact base: OpenCV 4.13.0 commit b4c5ec4042f097e2a5b386b9d413ec7333d0a184.
 * Changes: source/target membership data gate, membership-gated horizontal/vertical smoothness,
 *          source-active SOR support. Pressure diffusion is NOT included.
 */

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "amv_vr_masked_factory.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <chrono>

using namespace std;

namespace
{
using AMVProfileClock = std::chrono::steady_clock;

inline double elapsedMilliseconds(const AMVProfileClock::time_point &started)
{
    return std::chrono::duration<double, std::milli>(
               AMVProfileClock::now() - started)
        .count();
}
} // namespace

namespace cv
{

class AMVMaskedVariationalRefinementImpl CV_FINAL : public AMVMaskedVariationalRefinement
{
  public:
    AMVMaskedVariationalRefinementImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) CV_OVERRIDE;
    void calcMasked(InputArray I0, InputArray I1, InputArray source_membership, InputArray target_membership,
                    InputOutputArray flow, float active_threshold, float target_invalid_weight) CV_OVERRIDE;
    void calcUVMasked(InputArray I0, InputArray I1, InputArray source_membership, InputArray target_membership,
                      InputOutputArray flow_u, InputOutputArray flow_v, float active_threshold,
                      float target_invalid_weight) CV_OVERRIDE;
    AMVMaskedVRProfile getLastMaskedProfile() const CV_OVERRIDE { return lastMaskedProfile; }
    void collectGarbage() CV_OVERRIDE;

  protected: //!< algorithm parameters
    int fixedPointIterations, sorIterations;
    float omega;
    float alpha, delta, gamma;
    float zeta, epsilon;

  public:
    int getFixedPointIterations() const CV_OVERRIDE { return fixedPointIterations; }
    void setFixedPointIterations(int val) CV_OVERRIDE { fixedPointIterations = val; }
    int getSorIterations() const CV_OVERRIDE { return sorIterations; }
    void setSorIterations(int val) CV_OVERRIDE { sorIterations = val; }
    float getOmega() const CV_OVERRIDE { return omega; }
    void setOmega(float val) CV_OVERRIDE { omega = val; }
    float getAlpha() const CV_OVERRIDE { return alpha; }
    void setAlpha(float val) CV_OVERRIDE { alpha = val; }
    float getDelta() const CV_OVERRIDE { return delta; }
    void setDelta(float val) CV_OVERRIDE { delta = val; }
    float getGamma() const CV_OVERRIDE { return gamma; }
    void setGamma(float val) CV_OVERRIDE { gamma = val; }
    float getEpsilon() const CV_OVERRIDE { return epsilon; }
    void setEpsilon(float val) CV_OVERRIDE { epsilon = val; }
    bool getSparseMaskedSorEnabled() const CV_OVERRIDE { return sparseMaskedSorEnabled; }
    void setSparseMaskedSorEnabled(bool val) CV_OVERRIDE { sparseMaskedSorEnabled = val; }

  protected: //!< internal buffers
    /* This struct defines a special data layout for Mat_<float>. Original buffer is split into two: one for "red"
     * elements (sum of indices is even) and one for "black" (sum of indices is odd) in a checkerboard pattern. It
     * allows for more efficient processing in SOR iterations, more natural SIMD vectorization and parallelization
     * (Red-Black SOR). Additionally, it simplifies border handling by adding repeated borders to both red and
     * black buffers.
     */
    struct RedBlackBuffer
    {
        Mat_<float> red;   //!< (i+j)%2==0
        Mat_<float> black; //!< (i+j)%2==1

        /* Width of even and odd rows may be different */
        int red_even_len, red_odd_len;
        int black_even_len, black_odd_len;

        RedBlackBuffer();
        void create(Size s);
        void release();
    };

    Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;                            //!< image derivative buffers
    RedBlackBuffer Ix_rb, Iy_rb, Iz_rb, Ixx_rb, Ixy_rb, Iyy_rb, Ixz_rb, Iyz_rb; //!< corresponding red-black buffers

    RedBlackBuffer A11, A12, A22, b1, b2; //!< main linear system coefficients
    RedBlackBuffer weights;               //!< base smoothness term weights in the current fixed point iteration

    // Masked AMV path only. The unmasked path remains unchanged.
    RedBlackBuffer sourceMembership_rb, activeMask_rb, dataGate_rb;
    RedBlackBuffer weightsHor, weightsVert;
    Mat_<float> sourceMembership, targetMembership, warpedTargetMembership, dataGate;
    Mat_<float> currentW_u, currentW_v, activeMaskFloat;
    Mat activeMask8;
    float activeThreshold, targetInvalidWeight;
    bool sparseMaskedSorEnabled;
    vector<int> activeBeginRed, activeEndRed;
    vector<int> activeBeginBlack, activeEndBlack;
    AMVMaskedVRProfile lastMaskedProfile;

    Mat_<float> mapX, mapY; //!< auxiliary buffers for remapping

    RedBlackBuffer tempW_u, tempW_v; //!< flow buffers that are modified in each fixed point iteration
    RedBlackBuffer dW_u, dW_v;       //!< optical flow increment
    RedBlackBuffer W_u_rb, W_v_rb;   //!< red-black-buffer version of the input flow

  private: //!< private methods and parallel sections
    void splitCheckerboard(RedBlackBuffer &dst, Mat &src);
    void mergeCheckerboard(Mat &dst, RedBlackBuffer &src);
    void updateRepeatedBorders(RedBlackBuffer &dst);
    void warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v);
    void warpMembership(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v);
    void prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v);
    void prepareMaskedBuffers(Mat &source_membership, Mat &target_membership, Mat &W_u, Mat &W_v,
                              float active_threshold, float target_invalid_weight);
    void buildActiveRowBounds();
    void updateMaskedDataGate();

    /* Parallelizing arbitrary operations with 3 input/output arguments */
    typedef void (AMVMaskedVariationalRefinementImpl::*Op)(void *op1, void *op2, void *op3);
    struct ParallelOp_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        vector<Op> ops;
        vector<void *> op1s;
        vector<void *> op2s;
        vector<void *> op3s;

        ParallelOp_ParBody(AMVMaskedVariationalRefinementImpl &_var, vector<Op> _ops, vector<void *> &_op1s,
                           vector<void *> &_op2s, vector<void *> &_op3s);
        void operator()(const Range &range) const CV_OVERRIDE;
    };
    void gradHorizAndSplitOp(void *src, void *dst, void *dst_split)
    {
        CV_INSTRUMENT_REGION();

        Sobel(*(Mat *)src, *(Mat *)dst, -1, 1, 0, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
    }
    void gradVertAndSplitOp(void *src, void *dst, void *dst_split)
    {
        CV_INSTRUMENT_REGION();

        Sobel(*(Mat *)src, *(Mat *)dst, -1, 0, 1, 1, 1, 0.00, BORDER_REPLICATE);
        splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
    }
    void averageOp(void *src1, void *src2, void *dst)
    {
        CV_INSTRUMENT_REGION();

        addWeighted(*(Mat *)src1, 0.5, *(Mat *)src2, 0.5, 0.0, *(Mat *)dst, CV_32F);
    }
    void subtractOp(void *src1, void *src2, void *dst)
    {
        CV_INSTRUMENT_REGION();

        subtract(*(Mat *)src1, *(Mat *)src2, *(Mat *)dst, noArray(), CV_32F);
    }

    struct ComputeDataTerm_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        ComputeDataTerm_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                                RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeSmoothnessTermHorPass_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
        bool red_pass;

        ComputeSmoothnessTermHorPass_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h,
                                             RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, RedBlackBuffer &_tempW_u,
                                             RedBlackBuffer &_tempW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeSmoothnessTermVertPass_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v;
        bool red_pass;

        ComputeSmoothnessTermVertPass_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h,
                                              RedBlackBuffer &W_u, RedBlackBuffer &_W_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeMaskedDataTerm_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        ComputeMaskedDataTerm_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h,
                                      RedBlackBuffer &_dW_u, RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeMaskedSmoothnessTermHorPass_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
        bool red_pass;

        ComputeMaskedSmoothnessTermHorPass_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes,
                                                   int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
                                                   RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v,
                                                   bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct ComputeMaskedSmoothnessTermVertPass_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *W_u, *W_v;
        bool red_pass;

        ComputeMaskedSmoothnessTermVertPass_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes,
                                                    int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
                                                    bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct MaskedRedBlackSOR_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        MaskedRedBlackSOR_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h,
                                  RedBlackBuffer &_dW_u, RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };

    struct RedBlackSOR_ParBody : public ParallelLoopBody
    {
        AMVMaskedVariationalRefinementImpl *var;
        int nstripes, stripe_sz;
        int h;
        RedBlackBuffer *dW_u, *dW_v;
        bool red_pass;

        RedBlackSOR_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
                            RedBlackBuffer &_dW_v, bool _red_pass);
        void operator()(const Range &range) const CV_OVERRIDE;
    };
};

AMVMaskedVariationalRefinementImpl::AMVMaskedVariationalRefinementImpl()
{
    CV_INSTRUMENT_REGION();

    fixedPointIterations = 5;
    sorIterations = 5;
    alpha = 20.0f;
    delta = 5.0f;
    gamma = 10.0f;
    omega = 1.6f;
    zeta = 0.1f;
    epsilon = 0.001f;
    activeThreshold = 0.01f;
    targetInvalidWeight = 0.0f;
    sparseMaskedSorEnabled = true;
}

/* This function converts an input Mat into the RedBlackBuffer format, which involves
 * splitting the input buffer into two and adding repeated borders. Assumes that enough
 * memory in dst is already allocated.
 */
void AMVMaskedVariationalRefinementImpl::splitCheckerboard(RedBlackBuffer &dst, Mat &src)
{
    CV_INSTRUMENT_REGION();

    int buf_j, j;
    int buf_w = (int)ceil(src.cols / 2.0) + 2; //!< max width of red/black buffers with borders

    /* Rows of red and black buffers can have different actual width, some extra repeated values are
     * added for padding in such cases.
     */
    for (int i = 0; i < src.rows; i++)
    {
        float *src_buf = src.ptr<float>(i);
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);
        r_buf[0] = b_buf[0] = src_buf[0];
        buf_j = 1;
        if (i % 2 == 0)
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                r_buf[buf_j] = src_buf[j];
                b_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        else
        {
            for (j = 0; j < src.cols - 1; j += 2)
            {
                b_buf[buf_j] = src_buf[j];
                r_buf[buf_j] = src_buf[j + 1];
                buf_j++;
            }
            if (j < src.cols)
                r_buf[buf_j] = b_buf[buf_j] = src_buf[j];
            else
                j--;
        }
        r_buf[buf_w - 1] = b_buf[buf_w - 1] = src_buf[j];
    }

    /* Fill top and bottom borders: */
    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

/* The inverse of splitCheckerboard, i.e. converting the RedBlackBuffer back into Mat.
 * Assumes that enough memory in dst is already allocated.
 */
void AMVMaskedVariationalRefinementImpl::mergeCheckerboard(Mat &dst, RedBlackBuffer &src)
{
    CV_INSTRUMENT_REGION();

    int buf_j, j;
    for (int i = 0; i < dst.rows; i++)
    {
        float *src_r_buf = src.red.ptr<float>(i + 1);
        float *src_b_buf = src.black.ptr<float>(i + 1);
        float *dst_buf = dst.ptr<float>(i);
        buf_j = 1;

        if (i % 2 == 0)
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_r_buf[buf_j];
                dst_buf[j + 1] = src_b_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_r_buf[buf_j];
        }
        else
        {
            for (j = 0; j < dst.cols - 1; j += 2)
            {
                dst_buf[j] = src_b_buf[buf_j];
                dst_buf[j + 1] = src_r_buf[buf_j];
                buf_j++;
            }
            if (j < dst.cols)
                dst_buf[j] = src_b_buf[buf_j];
        }
    }
}

/* An auxiliary function that updates the borders. Used to enforce that border values repeat
 * the ones adjacent to the border.
 */
void AMVMaskedVariationalRefinementImpl::updateRepeatedBorders(RedBlackBuffer &dst)
{
    CV_INSTRUMENT_REGION();

    int buf_w = dst.red.cols;
    for (int i = 0; i < dst.red.rows - 2; i++)
    {
        float *r_buf = dst.red.ptr<float>(i + 1);
        float *b_buf = dst.black.ptr<float>(i + 1);

        if (i % 2 == 0)
        {
            b_buf[0] = r_buf[1];
            if (dst.red_even_len > dst.black_even_len)
                b_buf[dst.black_even_len + 1] = r_buf[dst.red_even_len];
            else
                r_buf[dst.red_even_len + 1] = b_buf[dst.black_even_len];
        }
        else
        {
            r_buf[0] = b_buf[1];
            if (dst.red_odd_len < dst.black_odd_len)
                r_buf[dst.red_odd_len + 1] = b_buf[dst.black_odd_len];
            else
                b_buf[dst.black_odd_len + 1] = r_buf[dst.red_odd_len];
        }
    }
    {
        float *r_buf_border = dst.red.ptr<float>(dst.red.rows - 1);
        float *b_buf_border = dst.black.ptr<float>(dst.black.rows - 1);
        float *r_buf = dst.red.ptr<float>(dst.red.rows - 2);
        float *b_buf = dst.black.ptr<float>(dst.black.rows - 2);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
    {
        float *r_buf_border = dst.red.ptr<float>(0);
        float *b_buf_border = dst.black.ptr<float>(0);
        float *r_buf = dst.red.ptr<float>(1);
        float *b_buf = dst.black.ptr<float>(1);
        memcpy(r_buf_border, b_buf, buf_w * sizeof(float));
        memcpy(b_buf_border, r_buf, buf_w * sizeof(float));
    }
}

AMVMaskedVariationalRefinementImpl::RedBlackBuffer::RedBlackBuffer()
{
    CV_INSTRUMENT_REGION();

    release();
}
void AMVMaskedVariationalRefinementImpl::RedBlackBuffer::create(Size s)
{
    CV_INSTRUMENT_REGION();

    /* Allocate enough memory to include borders */
    int w = (int)ceil(s.width / 2.0) + 2;
    red.create(s.height + 2, w);
    black.create(s.height + 2, w);

    if (s.width % 2 == 0)
        red_even_len = red_odd_len = black_even_len = black_odd_len = w - 2;
    else
    {
        red_even_len = black_odd_len = w - 2;
        red_odd_len = black_even_len = w - 3;
    }
}

void AMVMaskedVariationalRefinementImpl::RedBlackBuffer::release()
{
    CV_INSTRUMENT_REGION();

    red.release();
    black.release();
    red_even_len = red_odd_len = black_even_len = black_odd_len = 0;
}

AMVMaskedVariationalRefinementImpl::ParallelOp_ParBody::ParallelOp_ParBody(AMVMaskedVariationalRefinementImpl &_var, vector<Op> _ops,
                                                                  vector<void *> &_op1s, vector<void *> &_op2s,
                                                                  vector<void *> &_op3s)
    : var(&_var), ops(_ops), op1s(_op1s), op2s(_op2s), op3s(_op3s)
{
}

void AMVMaskedVariationalRefinementImpl::ParallelOp_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    for (int i = range.start; i < range.end; i++)
        (var->*ops[i])(op1s[i], op2s[i], op3s[i]);
}

void AMVMaskedVariationalRefinementImpl::warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < flow_u.rows; i++)
    {
        float *pFlowU = flow_u.ptr<float>(i);
        float *pFlowV = flow_v.ptr<float>(i);
        float *pMapX = mapX.ptr<float>(i);
        float *pMapY = mapY.ptr<float>(i);
        for (int j = 0; j < flow_u.cols; j++)
        {
            pMapX[j] = j + pFlowU[j];
            pMapY[j] = i + pFlowV[j];
        }
    }
    remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_REPLICATE);
}


void AMVMaskedVariationalRefinementImpl::warpMembership(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v)
{
    CV_INSTRUMENT_REGION();

    for (int i = 0; i < flow_u.rows; i++)
    {
        float *pFlowU = flow_u.ptr<float>(i);
        float *pFlowV = flow_v.ptr<float>(i);
        float *pMapX = mapX.ptr<float>(i);
        float *pMapY = mapY.ptr<float>(i);
        for (int j = 0; j < flow_u.cols; j++)
        {
            pMapX[j] = j + pFlowU[j];
            pMapY[j] = i + pFlowV[j];
        }
    }
    remap(src, dst, mapX, mapY, INTER_LINEAR, BORDER_CONSTANT, Scalar(0.0f));
}

void AMVMaskedVariationalRefinementImpl::prepareMaskedBuffers(Mat &source_membership, Mat &target_membership,
                                                               Mat &W_u, Mat &W_v, float active_threshold,
                                                               float target_invalid_weight)
{
    CV_INSTRUMENT_REGION();

    const auto started = AMVProfileClock::now();
    Size s = source_membership.size();
    activeThreshold = active_threshold;
    targetInvalidWeight = target_invalid_weight;

    /* The C API call is synchronous and the Python arrays remain alive for the
     * complete call. Keep shallow Mat headers instead of copying two full
     * membership images on every pyramid level and pressure state.
     */
    sourceMembership = Mat_<float>(source_membership);
    targetMembership = Mat_<float>(target_membership);

    sourceMembership_rb.create(s);
    activeMask_rb.create(s);
    dataGate_rb.create(s);
    weightsHor.create(s);
    weightsVert.create(s);
    weightsHor.red.setTo(0.0f);
    weightsHor.black.setTo(0.0f);
    weightsVert.red.setTo(0.0f);
    weightsVert.black.setTo(0.0f);

    currentW_u.create(s);
    currentW_v.create(s);
    warpedTargetMembership.create(s);
    dataGate.create(s);
    activeMaskFloat.create(s);
    activeMask8.create(s, CV_8U);

    compare(sourceMembership, Scalar(activeThreshold), activeMask8, CMP_GE);
    activeMask8.convertTo(activeMaskFloat, CV_32F, 1.0 / 255.0);

    W_u.setTo(0.0f, activeMask8 == 0);
    W_v.setTo(0.0f, activeMask8 == 0);

    splitCheckerboard(sourceMembership_rb, sourceMembership);
    splitCheckerboard(activeMask_rb, activeMaskFloat);
    buildActiveRowBounds();
    lastMaskedProfile.prepare_masked_buffers_ms += elapsedMilliseconds(started);
}

void AMVMaskedVariationalRefinementImpl::buildActiveRowBounds()
{
    CV_INSTRUMENT_REGION();

    const int height = activeMaskFloat.rows;
    activeBeginRed.assign(height, 0);
    activeEndRed.assign(height, 0);
    activeBeginBlack.assign(height, 0);
    activeEndBlack.assign(height, 0);

    for (int i = 0; i < height; ++i)
    {
        const int redLen = (i % 2 == 0)
                               ? activeMask_rb.red_even_len
                               : activeMask_rb.red_odd_len;
        const int blackLen = (i % 2 == 0)
                                 ? activeMask_rb.black_even_len
                                 : activeMask_rb.black_odd_len;

        const float *red = activeMask_rb.red.ptr<float>(i + 1) + 1;
        const float *black = activeMask_rb.black.ptr<float>(i + 1) + 1;

        int begin = 0;
        while (begin < redLen && red[begin] < 0.5f)
            ++begin;
        if (begin < redLen)
        {
            int end = redLen;
            while (end > begin && red[end - 1] < 0.5f)
                --end;
            activeBeginRed[i] = begin;
            activeEndRed[i] = end;
        }

        begin = 0;
        while (begin < blackLen && black[begin] < 0.5f)
            ++begin;
        if (begin < blackLen)
        {
            int end = blackLen;
            while (end > begin && black[end - 1] < 0.5f)
                --end;
            activeBeginBlack[i] = begin;
            activeEndBlack[i] = end;
        }
    }
}

void AMVMaskedVariationalRefinementImpl::updateMaskedDataGate()
{
    CV_INSTRUMENT_REGION();

    mergeCheckerboard(currentW_u, tempW_u);
    mergeCheckerboard(currentW_v, tempW_v);
    warpMembership(warpedTargetMembership, targetMembership, currentW_u, currentW_v);
    if (targetInvalidWeight > 0.0f)
        max(warpedTargetMembership, targetInvalidWeight, warpedTargetMembership);
    multiply(sourceMembership, warpedTargetMembership, dataGate, 1.0, CV_32F);
    splitCheckerboard(dataGate_rb, dataGate);
}

void AMVMaskedVariationalRefinementImpl::prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v)
{
    CV_INSTRUMENT_REGION();

    const auto totalStarted = AMVProfileClock::now();
    const auto allocationStarted = AMVProfileClock::now();

    Size s = I0.size();
    A11.create(s);
    A12.create(s);
    A22.create(s);
    b1.create(s);
    b2.create(s);
    weights.create(s);
    weights.red.setTo(0.0f);
    weights.black.setTo(0.0f);
    tempW_u.create(s);
    tempW_v.create(s);
    dW_u.create(s);
    dW_v.create(s);
    W_u_rb.create(s);
    W_v_rb.create(s);

    Ix.create(s);
    Iy.create(s);
    Iz.create(s);
    Ixx.create(s);
    Ixy.create(s);
    Iyy.create(s);
    Ixz.create(s);
    Iyz.create(s);

    Ix_rb.create(s);
    Iy_rb.create(s);
    Iz_rb.create(s);
    Ixx_rb.create(s);
    Ixy_rb.create(s);
    Iyy_rb.create(s);
    Ixz_rb.create(s);
    Iyz_rb.create(s);

    mapX.create(s);
    mapY.create(s);
    lastMaskedProfile.prepare_allocations_ms += elapsedMilliseconds(allocationStarted);

    /* Floating point warps work significantly better than fixed-point.
     * The pressure-conditioned Python path already supplies CV_32F images, so
     * use a shallow Mat header instead of copying the complete image.
     */
    Mat I1flt;
    const auto conversionStarted = AMVProfileClock::now();
    if (I1.type() == CV_32FC1)
        I1flt = I1;
    else
        I1.convertTo(I1flt, CV_32F);
    lastMaskedProfile.prepare_i1_conversion_ms += elapsedMilliseconds(conversionStarted);

    Mat warpedI;
    const auto warpStarted = AMVProfileClock::now();
    warpImage(warpedI, I1flt, W_u, W_v);
    lastMaskedProfile.prepare_warp_ms += elapsedMilliseconds(warpStarted);

    /* Computing an average of the current and warped next frames (to compute the derivatives on) and
     * temporal derivative Iz
     */
    Mat averagedI;
    const auto averageStarted = AMVProfileClock::now();
    {
        vector<void *> op1s;
        op1s.push_back((void *)&I0);
        op1s.push_back((void *)&warpedI);
        vector<void *> op2s;
        op2s.push_back((void *)&warpedI);
        op2s.push_back((void *)&I0);
        vector<void *> op3s;
        op3s.push_back((void *)&averagedI);
        op3s.push_back((void *)&Iz);
        vector<Op> ops;
        ops.push_back(&AMVMaskedVariationalRefinementImpl::averageOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::subtractOp);
        parallel_for_(Range(0, 2), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }
    splitCheckerboard(Iz_rb, Iz);
    lastMaskedProfile.prepare_average_subtract_ms += elapsedMilliseconds(averageStarted);

    const auto derivativesStarted = AMVProfileClock::now();
    /* Computing first-order derivatives */
    {
        vector<void *> op1s;
        op1s.push_back((void *)&averagedI);
        op1s.push_back((void *)&averagedI);
        op1s.push_back((void *)&Iz);
        op1s.push_back((void *)&Iz);
        vector<void *> op2s;
        op2s.push_back((void *)&Ix);
        op2s.push_back((void *)&Iy);
        op2s.push_back((void *)&Ixz);
        op2s.push_back((void *)&Iyz);
        vector<void *> op3s;
        op3s.push_back((void *)&Ix_rb);
        op3s.push_back((void *)&Iy_rb);
        op3s.push_back((void *)&Ixz_rb);
        op3s.push_back((void *)&Iyz_rb);
        vector<Op> ops;
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 4), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }

    /* Computing second-order derivatives */
    {
        vector<void *> op1s;
        op1s.push_back((void *)&Ix);
        op1s.push_back((void *)&Ix);
        op1s.push_back((void *)&Iy);
        vector<void *> op2s;
        op2s.push_back((void *)&Ixx);
        op2s.push_back((void *)&Ixy);
        op2s.push_back((void *)&Iyy);
        vector<void *> op3s;
        op3s.push_back((void *)&Ixx_rb);
        op3s.push_back((void *)&Ixy_rb);
        op3s.push_back((void *)&Iyy_rb);
        vector<Op> ops;
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradHorizAndSplitOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradVertAndSplitOp);
        ops.push_back(&AMVMaskedVariationalRefinementImpl::gradVertAndSplitOp);
        parallel_for_(Range(0, 3), ParallelOp_ParBody(*this, ops, op1s, op2s, op3s));
    }
    lastMaskedProfile.prepare_derivatives_ms += elapsedMilliseconds(derivativesStarted);
    lastMaskedProfile.prepare_buffers_ms += elapsedMilliseconds(totalStarted);
}

AMVMaskedVariationalRefinementImpl::ComputeDataTerm_ParBody::ComputeDataTerm_ParBody(AMVMaskedVariationalRefinementImpl &_var,
                                                                            int _nstripes, int _h,
                                                                            RedBlackBuffer &_dW_u,
                                                                            RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function computes parts of the main linear system coefficients A11,A12,A22,b1,b1
 * that correspond to the data term, which includes color and gradient constancy assumptions.
 */
void AMVMaskedVariationalRefinementImpl::ComputeDataTerm_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float zeta_squared = var->zeta * var->zeta;
    float epsilon_squared = var->epsilon * var->epsilon;
    float gamma2 = var->gamma / 2;
    float delta2 = var->delta / 2;

    float *pIx, *pIy, *pIz;
    float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;

    float derivNorm, derivNorm2;
    float Ik1z, Ik1zx, Ik1zy;
    float weight;
    int len;
    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(color)                                                                                       \
    pIx = var->Ix_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIy = var->Iy_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIz = var->Iz_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIxx = var->Ixx_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxy = var->Ixy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyy = var->Iyy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxz = var->Ixz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyz = var->Iyz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pa11 = var->A11.color.ptr<float>(i + 1) + 1;                                                                       \
    pa12 = var->A12.color.ptr<float>(i + 1) + 1;                                                                       \
    pa22 = var->A22.color.ptr<float>(i + 1) + 1;                                                                       \
    pb1 = var->b1.color.ptr<float>(i + 1) + 1;                                                                         \
    pb2 = var->b2.color.ptr<float>(i + 1) + 1;                                                                         \
    pdU = dW_u->color.ptr<float>(i + 1) + 1;                                                                           \
    pdV = dW_v->color.ptr<float>(i + 1) + 1;                                                                           \
    if (i % 2 == 0)                                                                                                    \
        len = var->Ix_rb.color##_even_len;                                                                             \
    else                                                                                                               \
        len = var->Ix_rb.color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red);
        }
        else
        {
            INIT_ROW_POINTERS(black);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 zeta_vec = v_setall_f32(zeta_squared);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 delta_vec = v_setall_f32(delta2);
        v_float32x4 gamma_vec = v_setall_f32(gamma2);
        v_float32x4 zero_vec = v_setall_f32(0.0f);
        v_float32x4 pIx_vec, pIy_vec, pIz_vec, pdU_vec, pdV_vec;
        v_float32x4 pIxx_vec, pIxy_vec, pIyy_vec, pIxz_vec, pIyz_vec;
        v_float32x4 derivNorm_vec, derivNorm2_vec, weight_vec;
        v_float32x4 Ik1z_vec, Ik1zx_vec, Ik1zy_vec;
        v_float32x4 pa11_vec, pa12_vec, pa22_vec, pb1_vec, pb2_vec;

        for (; j < len - 3; j += 4)
        {
            pIx_vec = v_load(pIx + j);
            pIy_vec = v_load(pIy + j);
            pIz_vec = v_load(pIz + j);
            pdU_vec = v_load(pdU + j);
            pdV_vec = v_load(pdV + j);

            derivNorm_vec = v_add(v_add(v_mul(pIx_vec, pIx_vec), v_mul(pIy_vec, pIy_vec)), zeta_vec);
            Ik1z_vec = v_add(v_add(pIz_vec, v_mul(pIx_vec, pdU_vec)), v_mul(pIy_vec, pdV_vec));
            weight_vec = v_div(v_div(delta_vec, v_sqrt(v_add(v_div(v_mul(Ik1z_vec, Ik1z_vec), derivNorm_vec), eps_vec))), derivNorm_vec);

            pa11_vec = v_add(v_mul(weight_vec, v_mul(pIx_vec, pIx_vec)), zeta_vec);
            pa12_vec = v_mul(weight_vec, v_mul(pIx_vec, pIy_vec));
            pa22_vec = v_add(v_mul(weight_vec, v_mul(pIy_vec, pIy_vec)), zeta_vec);
            pb1_vec = v_sub(zero_vec, v_mul(weight_vec, v_mul(pIz_vec, pIx_vec)));
            pb2_vec = v_sub(zero_vec, v_mul(weight_vec, v_mul(pIz_vec, pIy_vec)));

            pIxx_vec = v_load(pIxx + j);
            pIxy_vec = v_load(pIxy + j);
            pIyy_vec = v_load(pIyy + j);
            pIxz_vec = v_load(pIxz + j);
            pIyz_vec = v_load(pIyz + j);

            derivNorm_vec = v_add(v_add(v_mul(pIxx_vec, pIxx_vec), v_mul(pIxy_vec, pIxy_vec)), zeta_vec);
            derivNorm2_vec = v_add(v_add(v_mul(pIyy_vec, pIyy_vec), v_mul(pIxy_vec, pIxy_vec)), zeta_vec);
            Ik1zx_vec = v_add(v_add(pIxz_vec, v_mul(pIxx_vec, pdU_vec)), v_mul(pIxy_vec, pdV_vec));
            Ik1zy_vec = v_add(v_add(pIyz_vec, v_mul(pIxy_vec, pdU_vec)), v_mul(pIyy_vec, pdV_vec));
            weight_vec = v_div(gamma_vec, v_sqrt(v_add(v_add(v_div(v_mul(Ik1zx_vec, Ik1zx_vec), derivNorm_vec), v_div(v_mul(Ik1zy_vec, Ik1zy_vec), derivNorm2_vec)), eps_vec)));

            pa11_vec = v_add(pa11_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxx_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIxy_vec), derivNorm2_vec))));
            pa12_vec = v_add(pa12_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxy_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIyy_vec), derivNorm2_vec))));
            pa22_vec = v_add(pa22_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxy_vec, pIxy_vec), derivNorm_vec), v_div(v_mul(pIyy_vec, pIyy_vec), derivNorm2_vec))));
            pb1_vec = v_sub(pb1_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxz_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIyz_vec), derivNorm2_vec))));
            pb2_vec = v_sub(pb2_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxy_vec, pIxz_vec), derivNorm_vec), v_div(v_mul(pIyy_vec, pIyz_vec), derivNorm2_vec))));

            v_store(pa11 + j, pa11_vec);
            v_store(pa12 + j, pa12_vec);
            v_store(pa22 + j, pa22_vec);
            v_store(pb1 + j, pb1_vec);
            v_store(pb2 + j, pb2_vec);
        }
#endif
        for (; j < len; j++)
        {
            /* Step 1: Compute color constancy terms */
            /* Normalization factor:*/
            derivNorm = pIx[j] * pIx[j] + pIy[j] * pIy[j] + zeta_squared;
            /* Color constancy penalty (computed by Taylor expansion):*/
            Ik1z = pIz[j] + pIx[j] * pdU[j] + pIy[j] * pdV[j];
            /* Weight of the color constancy term in the current fixed-point iteration divided by derivNorm: */
            weight = (delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared)) / derivNorm;
            /* Add respective color constancy terms to the linear system coefficients: */
            pa11[j] = weight * (pIx[j] * pIx[j]) + zeta_squared;
            pa12[j] = weight * (pIx[j] * pIy[j]);
            pa22[j] = weight * (pIy[j] * pIy[j]) + zeta_squared;
            pb1[j] = -weight * (pIz[j] * pIx[j]);
            pb2[j] = -weight * (pIz[j] * pIy[j]);

            /* Step 2: Compute gradient constancy terms */
            /* Normalization factor for x gradient: */
            derivNorm = pIxx[j] * pIxx[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Normalization factor for y gradient: */
            derivNorm2 = pIyy[j] * pIyy[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Gradient constancy penalties (computed by Taylor expansion): */
            Ik1zx = pIxz[j] + pIxx[j] * pdU[j] + pIxy[j] * pdV[j];
            Ik1zy = pIyz[j] + pIxy[j] * pdU[j] + pIyy[j] * pdV[j];
            /* Weight of the gradient constancy term in the current fixed-point iteration: */
            weight = gamma2 / sqrt(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + epsilon_squared);
            /* Add respective gradient constancy components to the linear system coefficients: */
            pa11[j] += weight * (pIxx[j] * pIxx[j] / derivNorm + pIxy[j] * pIxy[j] / derivNorm2);
            pa12[j] += weight * (pIxx[j] * pIxy[j] / derivNorm + pIxy[j] * pIyy[j] / derivNorm2);
            pa22[j] += weight * (pIxy[j] * pIxy[j] / derivNorm + pIyy[j] * pIyy[j] / derivNorm2);
            pb1[j] += -weight * (pIxx[j] * pIxz[j] / derivNorm + pIxy[j] * pIyz[j] / derivNorm2);
            pb2[j] += -weight * (pIxy[j] * pIxz[j] / derivNorm + pIyy[j] * pIyz[j] / derivNorm2);
        }
    }
}

AMVMaskedVariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::ComputeSmoothnessTermHorPass_ParBody(
  AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
  RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), W_u(&_W_u), W_v(&_W_v), curW_u(&_tempW_u), curW_v(&_tempW_v),
      red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function updates the linear system coefficients A11,A22,b1,b1 according to the
 * flow smoothness term and computes corresponding weights for the current fixed point iteration.
 * A11,A22,b1,b1 are updated only partially (horizontal pass). Doing both horizontal and vertical
 * passes in one loop complicates parallelization (different threads write to the same elements).
 */
void AMVMaskedVariationalRefinementImpl::ComputeSmoothnessTermHorPass_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float epsilon_squared = var->epsilon * var->epsilon;
    float alpha2 = var->alpha / 2;
    float *pWeight;
    float *pA_u, *pA_u_next, *pA_v, *pA_v_next;
    float *pB_u, *pB_u_next, *pB_v, *pB_v_next;
    float *cW_u, *cW_u_next, *cW_u_next_row;
    float *cW_v, *cW_v_next, *cW_v_next_row;
    float *pW_u, *pW_u_next;
    float *pW_v, *pW_v_next;
    float ux, uy, vx, vy;
    int len;
    bool touches_right_border = true;

#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd, bool_default)                          \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_u = curW_u->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_v = curW_v->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    cW_u_next_row = curW_u->next_color.ptr<float>(i + 2) + 1;                                                          \
    cW_v_next_row = curW_v->next_color.ptr<float>(i + 2) + 1;                                                          \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = bool_default;                                                                       \
        else                                                                                                           \
            touches_right_border = !bool_default;                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = !bool_default;                                                                      \
        else                                                                                                           \
            touches_right_border = bool_default;                                                                       \
    }

    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2, true);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1, false);
        }
#undef INIT_ROW_POINTERS

#define COMPUTE                                                                                                        \
    /* Gradients for the flow on the current fixed-point iteration: */                                                 \
    ux = cW_u_next[j] - cW_u[j];                                                                                       \
    vx = cW_v_next[j] - cW_v[j];                                                                                       \
    uy = cW_u_next_row[j] - cW_u[j];                                                                                   \
    vy = cW_v_next_row[j] - cW_v[j];                                                                                   \
    /* Weight of the smoothness term in the current fixed-point iteration: */                                          \
    pWeight[j] = alpha2 / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon_squared);                               \
    /* Gradients for initial raw flow multiplied by weight:*/                                                          \
    ux = pWeight[j] * (pW_u_next[j] - pW_u[j]);                                                                        \
    vx = pWeight[j] * (pW_v_next[j] - pW_v[j]);

#define UPDATE                                                                                                         \
    pB_u[j] += ux;                                                                                                     \
    pA_u[j] += pWeight[j];                                                                                             \
    pB_v[j] += vx;                                                                                                     \
    pA_v[j] += pWeight[j];                                                                                             \
    pB_u_next[j] -= ux;                                                                                                \
    pA_u_next[j] += pWeight[j];                                                                                        \
    pB_v_next[j] -= vx;                                                                                                \
    pA_v_next[j] += pWeight[j];

        int j = 0;
#if CV_SIMD128
        v_float32x4 alpha2_vec = v_setall_f32(alpha2);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 cW_u_vec, cW_v_vec;
        v_float32x4 pWeight_vec, ux_vec, vx_vec, uy_vec, vy_vec;

        for (; j < len - 4; j += 4)
        {
            cW_u_vec = v_load(cW_u + j);
            cW_v_vec = v_load(cW_v + j);

            ux_vec = v_sub(v_load(cW_u_next + j), cW_u_vec);
            vx_vec = v_sub(v_load(cW_v_next + j), cW_v_vec);
            uy_vec = v_sub(v_load(cW_u_next_row + j), cW_u_vec);
            vy_vec = v_sub(v_load(cW_v_next_row + j), cW_v_vec);
            pWeight_vec =
              v_div(alpha2_vec, v_sqrt(v_add(v_add(v_add(v_add(v_mul(ux_vec, ux_vec), v_mul(vx_vec, vx_vec)), v_mul(uy_vec, uy_vec)), v_mul(vy_vec, vy_vec)), eps_vec)));
            v_store(pWeight + j, pWeight_vec);

            ux_vec = v_mul(pWeight_vec, v_sub(v_load(pW_u_next + j), v_load(pW_u + j)));
            vx_vec = v_mul(pWeight_vec, v_sub(v_load(pW_v_next + j), v_load(pW_v + j)));

            v_store(pA_u + j, v_add(v_load(pA_u + j), pWeight_vec));
            v_store(pA_v + j, v_add(v_load(pA_v + j), pWeight_vec));
            v_store(pB_u + j, v_add(v_load(pB_u + j), ux_vec));
            v_store(pB_v + j, v_add(v_load(pB_v + j), vx_vec));

            v_store(pA_u_next + j, v_add(v_load(pA_u_next + j), pWeight_vec));
            v_store(pA_v_next + j, v_add(v_load(pA_v_next + j), pWeight_vec));
            v_store(pB_u_next + j, v_sub(v_load(pB_u_next + j), ux_vec));
            v_store(pB_v_next + j, v_sub(v_load(pB_v_next + j), vx_vec));
        }
#endif
        for (; j < len - 1; j++)
        {
            COMPUTE;
            UPDATE;
        }

        /* Omit the update on the rightmost elements */
        if (touches_right_border)
        {
            COMPUTE;
        }
        else
        {
            COMPUTE;
            UPDATE;
        }
    }
#undef COMPUTE
#undef UPDATE
}

AMVMaskedVariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::ComputeSmoothnessTermVertPass_ParBody(
  AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), W_u(&_W_u), W_v(&_W_v), red_pass(_red_pass)
{
    /* Omit the last row in the vertical pass */
    h = _h - 1;
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function adds the last remaining terms to the linear system coefficients A11,A22,b1,b1. */
void AMVMaskedVariationalRefinementImpl::ComputeSmoothnessTermVertPass_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float *pWeight;
    float *pA_u, *pA_u_next_row, *pA_v, *pA_v_next_row;
    float *pB_u, *pB_u_next_row, *pB_v, *pB_v_next_row;
    float *pW_u, *pW_u_next_row, *pW_v, *pW_v_next_row;
    float vy, uy;
    int len;

    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color)                                                                       \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pA_u_next_row = var->A11.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_u_next_row = var->b1.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_u_next_row = W_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pA_v_next_row = var->A22.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_v_next_row = var->b2.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_v_next_row = W_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
        len = var->A11.cur_color##_even_len;                                                                           \
    else                                                                                                               \
        len = var->A11.cur_color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black);
        }
        else
        {
            INIT_ROW_POINTERS(black, red);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 pWeight_vec, uy_vec, vy_vec;
        for (; j < len - 3; j += 4)
        {
            pWeight_vec = v_load(pWeight + j);
            uy_vec = v_mul(pWeight_vec, v_sub(v_load(pW_u_next_row + j), v_load(pW_u + j)));
            vy_vec = v_mul(pWeight_vec, v_sub(v_load(pW_v_next_row + j), v_load(pW_v + j)));

            v_store(pA_u + j, v_add(v_load(pA_u + j), pWeight_vec));
            v_store(pA_v + j, v_add(v_load(pA_v + j), pWeight_vec));
            v_store(pB_u + j, v_add(v_load(pB_u + j), uy_vec));
            v_store(pB_v + j, v_add(v_load(pB_v + j), vy_vec));

            v_store(pA_u_next_row + j, v_add(v_load(pA_u_next_row + j), pWeight_vec));
            v_store(pA_v_next_row + j, v_add(v_load(pA_v_next_row + j), pWeight_vec));
            v_store(pB_u_next_row + j, v_sub(v_load(pB_u_next_row + j), uy_vec));
            v_store(pB_v_next_row + j, v_sub(v_load(pB_v_next_row + j), vy_vec));
        }
#endif
        for (; j < len; j++)
        {
            uy = pWeight[j] * (pW_u_next_row[j] - pW_u[j]);
            vy = pWeight[j] * (pW_v_next_row[j] - pW_v[j]);
            pB_u[j] += uy;
            pA_u[j] += pWeight[j];
            pB_v[j] += vy;
            pA_v[j] += pWeight[j];
            pB_u_next_row[j] -= uy;
            pA_u_next_row[j] += pWeight[j];
            pB_v_next_row[j] -= vy;
            pA_v_next_row[j] += pWeight[j];
        }
    }
}


AMVMaskedVariationalRefinementImpl::ComputeMaskedDataTerm_ParBody::ComputeMaskedDataTerm_ParBody(AMVMaskedVariationalRefinementImpl &_var,
                                                                            int _nstripes, int _h,
                                                                            RedBlackBuffer &_dW_u,
                                                                            RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void AMVMaskedVariationalRefinementImpl::ComputeMaskedDataTerm_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float zeta_squared = var->zeta * var->zeta;
    float epsilon_squared = var->epsilon * var->epsilon;
    float gamma2 = var->gamma / 2;
    float delta2 = var->delta / 2;

    float *pIx, *pIy, *pIz;
    float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
    float *pdU, *pdV;
    float *pa11, *pa12, *pa22, *pb1, *pb2;
    float *pGate, *pActive;

    float derivNorm, derivNorm2;
    float Ik1z, Ik1zx, Ik1zy;
    float weight;
    int len;
    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(color)                                                                                       \
    pIx = var->Ix_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIy = var->Iy_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIz = var->Iz_rb.color.ptr<float>(i + 1) + 1;                                                                      \
    pIxx = var->Ixx_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxy = var->Ixy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyy = var->Iyy_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIxz = var->Ixz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pIyz = var->Iyz_rb.color.ptr<float>(i + 1) + 1;                                                                    \
    pa11 = var->A11.color.ptr<float>(i + 1) + 1;                                                                       \
    pa12 = var->A12.color.ptr<float>(i + 1) + 1;                                                                       \
    pa22 = var->A22.color.ptr<float>(i + 1) + 1;                                                                       \
    pb1 = var->b1.color.ptr<float>(i + 1) + 1;                                                                         \
    pb2 = var->b2.color.ptr<float>(i + 1) + 1;                                                                         \
    pGate = var->dataGate_rb.color.ptr<float>(i + 1) + 1;                                                              \
    pActive = var->activeMask_rb.color.ptr<float>(i + 1) + 1;                                                          \
    pdU = dW_u->color.ptr<float>(i + 1) + 1;                                                                           \
    pdV = dW_v->color.ptr<float>(i + 1) + 1;                                                                           \
    if (i % 2 == 0)                                                                                                    \
        len = var->Ix_rb.color##_even_len;                                                                             \
    else                                                                                                               \
        len = var->Ix_rb.color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red);
        }
        else
        {
            INIT_ROW_POINTERS(black);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 zeta_vec = v_setall_f32(zeta_squared);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 delta_vec = v_setall_f32(delta2);
        v_float32x4 gamma_vec = v_setall_f32(gamma2);
        v_float32x4 zero_vec = v_setall_f32(0.0f);
        v_float32x4 pIx_vec, pIy_vec, pIz_vec, pdU_vec, pdV_vec;
        v_float32x4 pIxx_vec, pIxy_vec, pIyy_vec, pIxz_vec, pIyz_vec;
        v_float32x4 derivNorm_vec, derivNorm2_vec, weight_vec;
        v_float32x4 Ik1z_vec, Ik1zx_vec, Ik1zy_vec;
        v_float32x4 pa11_vec, pa12_vec, pa22_vec, pb1_vec, pb2_vec;
        v_float32x4 gate_vec, active_vec, inactive_vec;
        v_float32x4 one_vec = v_setall_f32(1.0f);
        v_float32x4 tiny_vec = v_setall_f32(1.0e-12f);

        for (; j < len - 3; j += 4)
        {
            pIx_vec = v_load(pIx + j);
            pIy_vec = v_load(pIy + j);
            pIz_vec = v_load(pIz + j);
            pdU_vec = v_load(pdU + j);
            pdV_vec = v_load(pdV + j);

            derivNorm_vec = v_add(v_add(v_mul(pIx_vec, pIx_vec), v_mul(pIy_vec, pIy_vec)), zeta_vec);
            Ik1z_vec = v_add(v_add(pIz_vec, v_mul(pIx_vec, pdU_vec)), v_mul(pIy_vec, pdV_vec));
            weight_vec = v_div(v_div(delta_vec, v_sqrt(v_add(v_div(v_mul(Ik1z_vec, Ik1z_vec), derivNorm_vec), eps_vec))), derivNorm_vec);

            pa11_vec = v_add(v_mul(weight_vec, v_mul(pIx_vec, pIx_vec)), zeta_vec);
            pa12_vec = v_mul(weight_vec, v_mul(pIx_vec, pIy_vec));
            pa22_vec = v_add(v_mul(weight_vec, v_mul(pIy_vec, pIy_vec)), zeta_vec);
            pb1_vec = v_sub(zero_vec, v_mul(weight_vec, v_mul(pIz_vec, pIx_vec)));
            pb2_vec = v_sub(zero_vec, v_mul(weight_vec, v_mul(pIz_vec, pIy_vec)));

            pIxx_vec = v_load(pIxx + j);
            pIxy_vec = v_load(pIxy + j);
            pIyy_vec = v_load(pIyy + j);
            pIxz_vec = v_load(pIxz + j);
            pIyz_vec = v_load(pIyz + j);

            derivNorm_vec = v_add(v_add(v_mul(pIxx_vec, pIxx_vec), v_mul(pIxy_vec, pIxy_vec)), zeta_vec);
            derivNorm2_vec = v_add(v_add(v_mul(pIyy_vec, pIyy_vec), v_mul(pIxy_vec, pIxy_vec)), zeta_vec);
            Ik1zx_vec = v_add(v_add(pIxz_vec, v_mul(pIxx_vec, pdU_vec)), v_mul(pIxy_vec, pdV_vec));
            Ik1zy_vec = v_add(v_add(pIyz_vec, v_mul(pIxy_vec, pdU_vec)), v_mul(pIyy_vec, pdV_vec));
            weight_vec = v_div(gamma_vec, v_sqrt(v_add(v_add(v_div(v_mul(Ik1zx_vec, Ik1zx_vec), derivNorm_vec), v_div(v_mul(Ik1zy_vec, Ik1zy_vec), derivNorm2_vec)), eps_vec)));

            pa11_vec = v_add(pa11_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxx_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIxy_vec), derivNorm2_vec))));
            pa12_vec = v_add(pa12_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxy_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIyy_vec), derivNorm2_vec))));
            pa22_vec = v_add(pa22_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxy_vec, pIxy_vec), derivNorm_vec), v_div(v_mul(pIyy_vec, pIyy_vec), derivNorm2_vec))));
            pb1_vec = v_sub(pb1_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxx_vec, pIxz_vec), derivNorm_vec), v_div(v_mul(pIxy_vec, pIyz_vec), derivNorm2_vec))));
            pb2_vec = v_sub(pb2_vec, v_mul(weight_vec, v_add(v_div(v_mul(pIxy_vec, pIxz_vec), derivNorm_vec), v_div(v_mul(pIyy_vec, pIyz_vec), derivNorm2_vec))));

            gate_vec = v_load(pGate + j);
            active_vec = v_load(pActive + j);
            inactive_vec = v_sub(one_vec, active_vec);
            pa11_vec = v_add(v_add(v_mul(pa11_vec, gate_vec), inactive_vec), v_mul(active_vec, tiny_vec));
            pa12_vec = v_mul(pa12_vec, gate_vec);
            pa22_vec = v_add(v_add(v_mul(pa22_vec, gate_vec), inactive_vec), v_mul(active_vec, tiny_vec));
            pb1_vec = v_mul(pb1_vec, gate_vec);
            pb2_vec = v_mul(pb2_vec, gate_vec);

            v_store(pa11 + j, pa11_vec);
            v_store(pa12 + j, pa12_vec);
            v_store(pa22 + j, pa22_vec);
            v_store(pb1 + j, pb1_vec);
            v_store(pb2 + j, pb2_vec);
        }
#endif
        for (; j < len; j++)
        {
            /* Step 1: Compute color constancy terms */
            /* Normalization factor:*/
            derivNorm = pIx[j] * pIx[j] + pIy[j] * pIy[j] + zeta_squared;
            /* Color constancy penalty (computed by Taylor expansion):*/
            Ik1z = pIz[j] + pIx[j] * pdU[j] + pIy[j] * pdV[j];
            /* Weight of the color constancy term in the current fixed-point iteration divided by derivNorm: */
            weight = (delta2 / sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared)) / derivNorm;
            /* Add respective color constancy terms to the linear system coefficients: */
            pa11[j] = weight * (pIx[j] * pIx[j]) + zeta_squared;
            pa12[j] = weight * (pIx[j] * pIy[j]);
            pa22[j] = weight * (pIy[j] * pIy[j]) + zeta_squared;
            pb1[j] = -weight * (pIz[j] * pIx[j]);
            pb2[j] = -weight * (pIz[j] * pIy[j]);

            /* Step 2: Compute gradient constancy terms */
            /* Normalization factor for x gradient: */
            derivNorm = pIxx[j] * pIxx[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Normalization factor for y gradient: */
            derivNorm2 = pIyy[j] * pIyy[j] + pIxy[j] * pIxy[j] + zeta_squared;
            /* Gradient constancy penalties (computed by Taylor expansion): */
            Ik1zx = pIxz[j] + pIxx[j] * pdU[j] + pIxy[j] * pdV[j];
            Ik1zy = pIyz[j] + pIxy[j] * pdU[j] + pIyy[j] * pdV[j];
            /* Weight of the gradient constancy term in the current fixed-point iteration: */
            weight = gamma2 / sqrt(Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2 + epsilon_squared);
            /* Add respective gradient constancy components to the linear system coefficients: */
            pa11[j] += weight * (pIxx[j] * pIxx[j] / derivNorm + pIxy[j] * pIxy[j] / derivNorm2);
            pa12[j] += weight * (pIxx[j] * pIxy[j] / derivNorm + pIxy[j] * pIyy[j] / derivNorm2);
            pa22[j] += weight * (pIxy[j] * pIxy[j] / derivNorm + pIyy[j] * pIyy[j] / derivNorm2);
            pb1[j] += -weight * (pIxx[j] * pIxz[j] / derivNorm + pIxy[j] * pIyz[j] / derivNorm2);
            pb2[j] += -weight * (pIxy[j] * pIxz[j] / derivNorm + pIyy[j] * pIyz[j] / derivNorm2);
            const float gate = pGate[j];
            const float active = pActive[j];
            const float inactive = 1.0f - active;
            pa11[j] = pa11[j] * gate + inactive + active * 1.0e-12f;
            pa12[j] *= gate;
            pa22[j] = pa22[j] * gate + inactive + active * 1.0e-12f;
            pb1[j] *= gate;
            pb2[j] *= gate;
        }
    }
}

AMVMaskedVariationalRefinementImpl::ComputeMaskedSmoothnessTermHorPass_ParBody::ComputeMaskedSmoothnessTermHorPass_ParBody(
  AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v,
  RedBlackBuffer &_tempW_u, RedBlackBuffer &_tempW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), W_u(&_W_u), W_v(&_W_v), curW_u(&_tempW_u), curW_v(&_tempW_v),
      red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void AMVMaskedVariationalRefinementImpl::ComputeMaskedSmoothnessTermHorPass_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float epsilon_squared = var->epsilon * var->epsilon;
    float alpha2 = var->alpha / 2;
    float *pWeight, *pWeightHor;
    float *pSource, *pSourceNext;
    float *pA_u, *pA_u_next, *pA_v, *pA_v_next;
    float *pB_u, *pB_u_next, *pB_v, *pB_v_next;
    float *cW_u, *cW_u_next, *cW_u_next_row;
    float *cW_v, *cW_v_next, *cW_v_next_row;
    float *pW_u, *pW_u_next;
    float *pW_v, *pW_v_next;
    float ux, uy, vx, vy;
    int len;
    bool touches_right_border = true;

#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd, bool_default)                          \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pWeightHor = var->weightsHor.cur_color.ptr<float>(i + 1) + 1;                                                      \
    pSource = var->sourceMembership_rb.cur_color.ptr<float>(i + 1) + 1;                                                \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_u = curW_u->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    cW_v = curW_v->cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    cW_u_next_row = curW_u->next_color.ptr<float>(i + 2) + 1;                                                          \
    cW_v_next_row = curW_v->next_color.ptr<float>(i + 2) + 1;                                                          \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pSourceNext = var->sourceMembership_rb.next_color.ptr<float>(i + 1) + next_offs_even;                          \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_even;                                            \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                             \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = bool_default;                                                                       \
        else                                                                                                           \
            touches_right_border = !bool_default;                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pA_u_next = var->A11.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_u_next = var->b1.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_u_next = curW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_u_next = W_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pSourceNext = var->sourceMembership_rb.next_color.ptr<float>(i + 1) + next_offs_odd;                          \
        pA_v_next = var->A22.next_color.ptr<float>(i + 1) + next_offs_odd;                                             \
        pB_v_next = var->b2.next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        cW_v_next = curW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                              \
        pW_v_next = W_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
        if (var->A11.cur_color##_even_len != var->A11.cur_color##_odd_len)                                             \
            touches_right_border = !bool_default;                                                                      \
        else                                                                                                           \
            touches_right_border = bool_default;                                                                       \
    }

    for (int i = start_i; i < end_i; i++)
    {
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2, true);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1, false);
        }
#undef INIT_ROW_POINTERS

#define COMPUTE                                                                                                        \
    /* Gradients for the flow on the current fixed-point iteration: */                                                 \
    ux = cW_u_next[j] - cW_u[j];                                                                                       \
    vx = cW_v_next[j] - cW_v[j];                                                                                       \
    uy = cW_u_next_row[j] - cW_u[j];                                                                                   \
    vy = cW_v_next_row[j] - cW_v[j];                                                                                   \
    /* Weight of the smoothness term in the current fixed-point iteration: */                                          \
    pWeight[j] = alpha2 / sqrt(ux * ux + vx * vx + uy * uy + vy * vy + epsilon_squared);                               \
    pWeightHor[j] = pWeight[j] * sqrt(max(0.0f, pSource[j] * pSourceNext[j]));                                          \
    /* Gradients for initial raw flow multiplied by masked horizontal edge weight:*/                                   \
    ux = pWeightHor[j] * (pW_u_next[j] - pW_u[j]);                                                                     \
    vx = pWeightHor[j] * (pW_v_next[j] - pW_v[j]);

#define UPDATE                                                                                                         \
    pB_u[j] += ux;                                                                                                     \
    pA_u[j] += pWeightHor[j];                                                                                             \
    pB_v[j] += vx;                                                                                                     \
    pA_v[j] += pWeightHor[j];                                                                                             \
    pB_u_next[j] -= ux;                                                                                                \
    pA_u_next[j] += pWeightHor[j];                                                                                        \
    pB_v_next[j] -= vx;                                                                                                \
    pA_v_next[j] += pWeightHor[j];

        int j = 0;
#if CV_SIMD128
        v_float32x4 alpha2_vec = v_setall_f32(alpha2);
        v_float32x4 eps_vec = v_setall_f32(epsilon_squared);
        v_float32x4 cW_u_vec, cW_v_vec;
        v_float32x4 pWeight_vec, pWeightHor_vec, ux_vec, vx_vec, uy_vec, vy_vec;
        v_float32x4 source_vec, source_next_vec;
        v_float32x4 zero_vec = v_setall_f32(0.0f);

        for (; j < len - 4; j += 4)
        {
            cW_u_vec = v_load(cW_u + j);
            cW_v_vec = v_load(cW_v + j);

            ux_vec = v_sub(v_load(cW_u_next + j), cW_u_vec);
            vx_vec = v_sub(v_load(cW_v_next + j), cW_v_vec);
            uy_vec = v_sub(v_load(cW_u_next_row + j), cW_u_vec);
            vy_vec = v_sub(v_load(cW_v_next_row + j), cW_v_vec);
            pWeight_vec =
              v_div(alpha2_vec, v_sqrt(v_add(v_add(v_add(v_add(v_mul(ux_vec, ux_vec), v_mul(vx_vec, vx_vec)), v_mul(uy_vec, uy_vec)), v_mul(vy_vec, vy_vec)), eps_vec)));
            v_store(pWeight + j, pWeight_vec);
            source_vec = v_load(pSource + j);
            source_next_vec = v_load(pSourceNext + j);
            pWeightHor_vec = v_mul(pWeight_vec, v_sqrt(v_max(v_mul(source_vec, source_next_vec), zero_vec)));
            v_store(pWeightHor + j, pWeightHor_vec);

            ux_vec = v_mul(pWeightHor_vec, v_sub(v_load(pW_u_next + j), v_load(pW_u + j)));
            vx_vec = v_mul(pWeightHor_vec, v_sub(v_load(pW_v_next + j), v_load(pW_v + j)));

            v_store(pA_u + j, v_add(v_load(pA_u + j), pWeightHor_vec));
            v_store(pA_v + j, v_add(v_load(pA_v + j), pWeightHor_vec));
            v_store(pB_u + j, v_add(v_load(pB_u + j), ux_vec));
            v_store(pB_v + j, v_add(v_load(pB_v + j), vx_vec));

            v_store(pA_u_next + j, v_add(v_load(pA_u_next + j), pWeightHor_vec));
            v_store(pA_v_next + j, v_add(v_load(pA_v_next + j), pWeightHor_vec));
            v_store(pB_u_next + j, v_sub(v_load(pB_u_next + j), ux_vec));
            v_store(pB_v_next + j, v_sub(v_load(pB_v_next + j), vx_vec));
        }
#endif
        for (; j < len - 1; j++)
        {
            COMPUTE;
            UPDATE;
        }

        /* Omit the update on the rightmost elements */
        if (touches_right_border)
        {
            COMPUTE;
        }
        else
        {
            COMPUTE;
            UPDATE;
        }
    }
#undef COMPUTE
#undef UPDATE
}

AMVMaskedVariationalRefinementImpl::ComputeMaskedSmoothnessTermVertPass_ParBody::ComputeMaskedSmoothnessTermVertPass_ParBody(
  AMVMaskedVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), W_u(&_W_u), W_v(&_W_v), red_pass(_red_pass)
{
    /* Omit the last row in the vertical pass */
    h = _h - 1;
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void AMVMaskedVariationalRefinementImpl::ComputeMaskedSmoothnessTermVertPass_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    float *pWeight, *pWeightVert;
    float *pSource, *pSourceNextRow;
    float *pA_u, *pA_u_next_row, *pA_v, *pA_v_next_row;
    float *pB_u, *pB_u_next_row, *pB_v, *pB_v_next_row;
    float *pW_u, *pW_u_next_row, *pW_v, *pW_v_next_row;
    float vy, uy;
    int len;

    for (int i = start_i; i < end_i; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color)                                                                       \
    pWeight = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pWeightVert = var->weightsVert.cur_color.ptr<float>(i + 1) + 1;                                                    \
    pSource = var->sourceMembership_rb.cur_color.ptr<float>(i + 1) + 1;                                                \
    pA_u = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_u = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_u = W_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pA_v = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pB_v = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                    \
    pW_v = W_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pA_u_next_row = var->A11.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_u_next_row = var->b1.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_u_next_row = W_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pSourceNextRow = var->sourceMembership_rb.next_color.ptr<float>(i + 2) + 1;                                       \
    pA_v_next_row = var->A22.next_color.ptr<float>(i + 2) + 1;                                                         \
    pB_v_next_row = var->b2.next_color.ptr<float>(i + 2) + 1;                                                          \
    pW_v_next_row = W_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
        len = var->A11.cur_color##_even_len;                                                                           \
    else                                                                                                               \
        len = var->A11.cur_color##_odd_len;

        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black);
        }
        else
        {
            INIT_ROW_POINTERS(black, red);
        }
#undef INIT_ROW_POINTERS

        int j = 0;
#if CV_SIMD128
        v_float32x4 pWeight_vec, pWeightVert_vec, uy_vec, vy_vec;
        v_float32x4 source_vec, source_next_row_vec;
        v_float32x4 zero_vec = v_setall_f32(0.0f);
        for (; j < len - 3; j += 4)
        {
            pWeight_vec = v_load(pWeight + j);
            source_vec = v_load(pSource + j);
            source_next_row_vec = v_load(pSourceNextRow + j);
            pWeightVert_vec = v_mul(pWeight_vec, v_sqrt(v_max(v_mul(source_vec, source_next_row_vec), zero_vec)));
            v_store(pWeightVert + j, pWeightVert_vec);
            uy_vec = v_mul(pWeightVert_vec, v_sub(v_load(pW_u_next_row + j), v_load(pW_u + j)));
            vy_vec = v_mul(pWeightVert_vec, v_sub(v_load(pW_v_next_row + j), v_load(pW_v + j)));

            v_store(pA_u + j, v_add(v_load(pA_u + j), pWeightVert_vec));
            v_store(pA_v + j, v_add(v_load(pA_v + j), pWeightVert_vec));
            v_store(pB_u + j, v_add(v_load(pB_u + j), uy_vec));
            v_store(pB_v + j, v_add(v_load(pB_v + j), vy_vec));

            v_store(pA_u_next_row + j, v_add(v_load(pA_u_next_row + j), pWeightVert_vec));
            v_store(pA_v_next_row + j, v_add(v_load(pA_v_next_row + j), pWeightVert_vec));
            v_store(pB_u_next_row + j, v_sub(v_load(pB_u_next_row + j), uy_vec));
            v_store(pB_v_next_row + j, v_sub(v_load(pB_v_next_row + j), vy_vec));
        }
#endif
        for (; j < len; j++)
        {
            pWeightVert[j] = pWeight[j] * sqrt(max(0.0f, pSource[j] * pSourceNextRow[j]));
            uy = pWeightVert[j] * (pW_u_next_row[j] - pW_u[j]);
            vy = pWeightVert[j] * (pW_v_next_row[j] - pW_v[j]);
            pB_u[j] += uy;
            pA_u[j] += pWeightVert[j];
            pB_v[j] += vy;
            pA_v[j] += pWeightVert[j];
            pB_u_next_row[j] -= uy;
            pA_u_next_row[j] += pWeightVert[j];
            pB_v_next_row[j] -= vy;
            pA_v_next_row[j] += pWeightVert[j];
        }
    }
}

AMVMaskedVariationalRefinementImpl::MaskedRedBlackSOR_ParBody::MaskedRedBlackSOR_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes,
                                                                    int _h, RedBlackBuffer &_dW_u,
                                                                    RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

void AMVMaskedVariationalRefinementImpl::MaskedRedBlackSOR_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);

    float *pa11, *pa12, *pa22, *pb1, *pb2, *pWH, *pWV, *pActive, *pdu, *pdv;
    float *pWH_next, *pdu_next, *pdv_next;
    float *pWV_prev_row, *pdu_prev_row, *pdv_prev_row;
    float *pdu_next_row, *pdv_next_row;

    float sigmaU, sigmaV;
    int j, len;
    for (int i = start; i < end; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd)                                        \
    pWH = var->weightsHor.cur_color.ptr<float>(i + 1) + 1;                                                            \
    pWV = var->weightsVert.cur_color.ptr<float>(i + 1) + 1;                                                           \
    pActive = var->activeMask_rb.cur_color.ptr<float>(i + 1) + 1;                                                     \
    pa11 = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa12 = var->A12.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa22 = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pb1 = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pb2 = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pdu = dW_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pdv = dW_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pdu_next_row = dW_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pdv_next_row = dW_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    pWV_prev_row = var->weightsVert.next_color.ptr<float>(i) + 1;                                                     \
    pdu_prev_row = dW_u->next_color.ptr<float>(i) + 1;                                                                 \
    pdv_prev_row = dW_v->next_color.ptr<float>(i) + 1;                                                                 \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pWH_next = var->weightsHor.next_color.ptr<float>(i + 1) + next_offs_even;                                          \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pWH_next = var->weightsHor.next_color.ptr<float>(i + 1) + next_offs_odd;                                           \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
    }
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1);
        }
#undef INIT_ROW_POINTERS

        int active_begin = 0;
        int active_end = len;
        if (var->sparseMaskedSorEnabled)
        {
            if (red_pass)
            {
                active_begin = var->activeBeginRed[i];
                active_end = var->activeEndRed[i];
            }
            else
            {
                active_begin = var->activeBeginBlack[i];
                active_end = var->activeEndBlack[i];
            }
            if (active_begin >= active_end)
                continue;
        }

        const int vectorized_end = len - (len % 4);
        int simd_begin = 0;
        int simd_end = vectorized_end;
        if (var->sparseMaskedSorEnabled)
        {
            /* Keep the original four-lane block alignment. This preserves the
             * exact SIMD arithmetic for every active lane while omitting only
             * leading/trailing inactive blocks.
             */
            simd_begin = min(vectorized_end, active_begin & ~3);
            simd_end = min(vectorized_end, (active_end + 3) & ~3);
        }

        j = simd_begin;
#if CV_SIMD128
        v_float32x4 pWH_prev_vec = v_setall_f32(pWH_next[j - 1]);
        v_float32x4 pdu_prev_vec = v_setall_f32(pdu_next[j - 1]);
        v_float32x4 pdv_prev_vec = v_setall_f32(pdv_next[j - 1]);
        v_float32x4 omega_vec = v_setall_f32(var->omega);
        v_float32x4 pWH_vec, pWH_next_vec, pWV_vec, pWV_prev_row_vec, active_vec, inactive_vec;
        v_float32x4 one_vec = v_setall_f32(1.0f);
        v_float32x4 pdu_next_vec, pdu_prev_row_vec, pdu_next_row_vec;
        v_float32x4 pdv_next_vec, pdv_prev_row_vec, pdv_next_row_vec;
        v_float32x4 pWH_shifted_vec, pdu_shifted_vec, pdv_shifted_vec;
        v_float32x4 pa12_vec, sigmaU_vec, sigmaV_vec, pdu_vec, pdv_vec;
        for (; j < simd_end; j += 4)
        {
            pWH_next_vec = v_load(pWH_next + j);
            pdu_next_vec = v_load(pdu_next + j);
            pdv_next_vec = v_load(pdv_next + j);

            if (var->sparseMaskedSorEnabled &&
                pActive[j] < 0.5f && pActive[j + 1] < 0.5f &&
                pActive[j + 2] < 0.5f && pActive[j + 3] < 0.5f)
            {
                /* These lanes are mathematically frozen by the source-active
                 * mask. Retain only the three values needed to construct the
                 * shifted neighbour vector in the following SIMD block.
                 */
                pWH_prev_vec = pWH_next_vec;
                pdu_prev_vec = pdu_next_vec;
                pdv_prev_vec = pdv_next_vec;
                continue;
            }

            pWH_vec = v_load(pWH + j);
            pWV_vec = v_load(pWV + j);
            pWV_prev_row_vec = v_load(pWV_prev_row + j);
            active_vec = v_load(pActive + j);
            inactive_vec = v_sub(one_vec, active_vec);
            pdu_prev_row_vec = v_load(pdu_prev_row + j);
            pdu_next_row_vec = v_load(pdu_next_row + j);
            pdv_prev_row_vec = v_load(pdv_prev_row + j);
            pdv_next_row_vec = v_load(pdv_next_row + j);
            pa12_vec = v_load(pa12 + j);
            pWH_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pWH_prev_vec), v_reinterpret_as_s32(pWH_next_vec)));
            pdu_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdu_prev_vec), v_reinterpret_as_s32(pdu_next_vec)));
            pdv_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdv_prev_vec), v_reinterpret_as_s32(pdv_next_vec)));

            sigmaU_vec = v_add(v_add(v_add(v_mul(pWH_shifted_vec, pdu_shifted_vec), v_mul(pWH_vec, pdu_next_vec)), v_mul(pWV_prev_row_vec, pdu_prev_row_vec)), v_mul(pWV_vec, pdu_next_row_vec));
            sigmaV_vec = v_add(v_add(v_add(v_mul(pWH_shifted_vec, pdv_shifted_vec), v_mul(pWH_vec, pdv_next_vec)), v_mul(pWV_prev_row_vec, pdv_prev_row_vec)), v_mul(pWV_vec, pdv_next_row_vec));

            v_float32x4 pdu_old_vec = v_load(pdu + j);
            v_float32x4 pdv_old_vec = v_load(pdv + j);
            pdu_vec = v_add(pdu_old_vec, v_mul(omega_vec, v_sub(v_div(v_sub(v_add(sigmaU_vec, v_load(pb1 + j)), v_mul(pdv_old_vec, pa12_vec)), v_load(pa11 + j)), pdu_old_vec)));
            pdu_vec = v_add(v_mul(active_vec, pdu_vec), v_mul(inactive_vec, pdu_old_vec));
            pdv_vec = v_add(pdv_old_vec, v_mul(omega_vec, v_sub(v_div(v_sub(v_add(sigmaV_vec, v_load(pb2 + j)), v_mul(pdu_vec, pa12_vec)), v_load(pa22 + j)), pdv_old_vec)));
            pdv_vec = v_add(v_mul(active_vec, pdv_vec), v_mul(inactive_vec, pdv_old_vec));
            v_store(pdu + j, pdu_vec);
            v_store(pdv + j, pdv_vec);

            pWH_prev_vec = pWH_next_vec;
            pdu_prev_vec = pdu_next_vec;
            pdv_prev_vec = pdv_next_vec;
        }
#endif
        if (var->sparseMaskedSorEnabled)
            j = max(j, active_begin);
        for (; j < active_end; j++)
        {
            if (var->sparseMaskedSorEnabled && pActive[j] < 0.5f)
                continue;

            sigmaU = pWH_next[j - 1] * pdu_next[j - 1] + pWH[j] * pdu_next[j] + pWV_prev_row[j] * pdu_prev_row[j] +
                     pWV[j] * pdu_next_row[j];
            sigmaV = pWH_next[j - 1] * pdv_next[j - 1] + pWH[j] * pdv_next[j] + pWV_prev_row[j] * pdv_prev_row[j] +
                     pWV[j] * pdv_next_row[j];
            if (pActive[j] >= 0.5f)
            {
                pdu[j] += var->omega * ((sigmaU + pb1[j] - pdv[j] * pa12[j]) / pa11[j] - pdu[j]);
                pdv[j] += var->omega * ((sigmaV + pb2[j] - pdu[j] * pa12[j]) / pa22[j] - pdv[j]);
            }
        }
    }
}

AMVMaskedVariationalRefinementImpl::RedBlackSOR_ParBody::RedBlackSOR_ParBody(AMVMaskedVariationalRefinementImpl &_var, int _nstripes,
                                                                    int _h, RedBlackBuffer &_dW_u,
                                                                    RedBlackBuffer &_dW_v, bool _red_pass)
    : var(&_var), nstripes(_nstripes), h(_h), dW_u(&_dW_u), dW_v(&_dW_v), red_pass(_red_pass)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function implements the Red-Black SOR (successive-over relaxation) method for solving the main
 * linear system in the current fixed-point iteration.
 */
void AMVMaskedVariationalRefinementImpl::RedBlackSOR_ParBody::operator()(const Range &range) const
{
    CV_INSTRUMENT_REGION();

    int start = min(range.start * stripe_sz, h);
    int end = min(range.end * stripe_sz, h);

    float *pa11, *pa12, *pa22, *pb1, *pb2, *pW, *pdu, *pdv;
    float *pW_next, *pdu_next, *pdv_next;
    float *pW_prev_row, *pdu_prev_row, *pdv_prev_row;
    float *pdu_next_row, *pdv_next_row;

    float sigmaU, sigmaV;
    int j, len;
    for (int i = start; i < end; i++)
    {
#define INIT_ROW_POINTERS(cur_color, next_color, next_offs_even, next_offs_odd)                                        \
    pW = var->weights.cur_color.ptr<float>(i + 1) + 1;                                                                 \
    pa11 = var->A11.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa12 = var->A12.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pa22 = var->A22.cur_color.ptr<float>(i + 1) + 1;                                                                   \
    pb1 = var->b1.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pb2 = var->b2.cur_color.ptr<float>(i + 1) + 1;                                                                     \
    pdu = dW_u->cur_color.ptr<float>(i + 1) + 1;                                                                       \
    pdv = dW_v->cur_color.ptr<float>(i + 1) + 1;                                                                       \
                                                                                                                       \
    pdu_next_row = dW_u->next_color.ptr<float>(i + 2) + 1;                                                             \
    pdv_next_row = dW_v->next_color.ptr<float>(i + 2) + 1;                                                             \
                                                                                                                       \
    pW_prev_row = var->weights.next_color.ptr<float>(i) + 1;                                                           \
    pdu_prev_row = dW_u->next_color.ptr<float>(i) + 1;                                                                 \
    pdv_prev_row = dW_v->next_color.ptr<float>(i) + 1;                                                                 \
                                                                                                                       \
    if (i % 2 == 0)                                                                                                    \
    {                                                                                                                  \
        pW_next = var->weights.next_color.ptr<float>(i + 1) + next_offs_even;                                          \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_even;                                                \
        len = var->A11.cur_color##_even_len;                                                                           \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pW_next = var->weights.next_color.ptr<float>(i + 1) + next_offs_odd;                                           \
        pdu_next = dW_u->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        pdv_next = dW_v->next_color.ptr<float>(i + 1) + next_offs_odd;                                                 \
        len = var->A11.cur_color##_odd_len;                                                                            \
    }
        if (red_pass)
        {
            INIT_ROW_POINTERS(red, black, 1, 2);
        }
        else
        {
            INIT_ROW_POINTERS(black, red, 2, 1);
        }
#undef INIT_ROW_POINTERS

        j = 0;
#if CV_SIMD128
        v_float32x4 pW_prev_vec = v_setall_f32(pW_next[-1]);
        v_float32x4 pdu_prev_vec = v_setall_f32(pdu_next[-1]);
        v_float32x4 pdv_prev_vec = v_setall_f32(pdv_next[-1]);
        v_float32x4 omega_vec = v_setall_f32(var->omega);
        v_float32x4 pW_vec, pW_next_vec, pW_prev_row_vec;
        v_float32x4 pdu_next_vec, pdu_prev_row_vec, pdu_next_row_vec;
        v_float32x4 pdv_next_vec, pdv_prev_row_vec, pdv_next_row_vec;
        v_float32x4 pW_shifted_vec, pdu_shifted_vec, pdv_shifted_vec;
        v_float32x4 pa12_vec, sigmaU_vec, sigmaV_vec, pdu_vec, pdv_vec;
        for (; j < len - 3; j += 4)
        {
            pW_vec = v_load(pW + j);
            pW_next_vec = v_load(pW_next + j);
            pW_prev_row_vec = v_load(pW_prev_row + j);
            pdu_next_vec = v_load(pdu_next + j);
            pdu_prev_row_vec = v_load(pdu_prev_row + j);
            pdu_next_row_vec = v_load(pdu_next_row + j);
            pdv_next_vec = v_load(pdv_next + j);
            pdv_prev_row_vec = v_load(pdv_prev_row + j);
            pdv_next_row_vec = v_load(pdv_next_row + j);
            pa12_vec = v_load(pa12 + j);
            pW_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pW_prev_vec), v_reinterpret_as_s32(pW_next_vec)));
            pdu_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdu_prev_vec), v_reinterpret_as_s32(pdu_next_vec)));
            pdv_shifted_vec = v_reinterpret_as_f32(
              v_extract<3>(v_reinterpret_as_s32(pdv_prev_vec), v_reinterpret_as_s32(pdv_next_vec)));

            sigmaU_vec = v_add(v_add(v_add(v_mul(pW_shifted_vec, pdu_shifted_vec), v_mul(pW_vec, pdu_next_vec)), v_mul(pW_prev_row_vec, pdu_prev_row_vec)), v_mul(pW_vec, pdu_next_row_vec));
            sigmaV_vec = v_add(v_add(v_add(v_mul(pW_shifted_vec, pdv_shifted_vec), v_mul(pW_vec, pdv_next_vec)), v_mul(pW_prev_row_vec, pdv_prev_row_vec)), v_mul(pW_vec, pdv_next_row_vec));

            pdu_vec = v_load(pdu + j);
            pdv_vec = v_load(pdv + j);
            pdu_vec = v_add(pdu_vec, v_mul(omega_vec, v_sub(v_div(v_sub(v_add(sigmaU_vec, v_load(pb1 + j)), v_mul(pdv_vec, pa12_vec)), v_load(pa11 + j)), pdu_vec)));
            pdv_vec = v_add(pdv_vec, v_mul(omega_vec, v_sub(v_div(v_sub(v_add(sigmaV_vec, v_load(pb2 + j)), v_mul(pdu_vec, pa12_vec)), v_load(pa22 + j)), pdv_vec)));
            v_store(pdu + j, pdu_vec);
            v_store(pdv + j, pdv_vec);

            pW_prev_vec = pW_next_vec;
            pdu_prev_vec = pdu_next_vec;
            pdv_prev_vec = pdv_next_vec;
        }
#endif
        for (; j < len; j++)
        {
            sigmaU = pW_next[j - 1] * pdu_next[j - 1] + pW[j] * pdu_next[j] + pW_prev_row[j] * pdu_prev_row[j] +
                     pW[j] * pdu_next_row[j];
            sigmaV = pW_next[j - 1] * pdv_next[j - 1] + pW[j] * pdv_next[j] + pW_prev_row[j] * pdv_prev_row[j] +
                     pW[j] * pdv_next_row[j];
            pdu[j] += var->omega * ((sigmaU + pb1[j] - pdv[j] * pa12[j]) / pa11[j] - pdu[j]);
            pdv[j] += var->omega * ((sigmaV + pb2[j] - pdu[j] * pa12[j]) / pa22[j] - pdv[j]);
        }
    }
}

void AMVMaskedVariationalRefinementImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) || (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!flow.empty() && flow.depth() == CV_32F && flow.channels() == 2);
    CV_Assert(I0.sameSize(flow));

    Mat uv[2];
    Mat &flowMat = flow.getMatRef();
    split(flowMat, uv);
    calcUV(I0, I1, uv[0], uv[1]);
    merge(uv, 2, flowMat);
}

void AMVMaskedVariationalRefinementImpl::calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) || (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!flow_u.empty() && flow_u.depth() == CV_32F && flow_u.channels() == 1);
    CV_Assert(!flow_v.empty() && flow_v.depth() == CV_32F && flow_v.channels() == 1);
    CV_Assert(I0.sameSize(flow_u));
    CV_Assert(flow_u.sameSize(flow_v));

    int num_stripes = getNumThreads();
    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    Mat &W_u = flow_u.getMatRef();
    Mat &W_v = flow_v.getMatRef();
    prepareBuffers(I0Mat, I1Mat, W_u, W_v);

    splitCheckerboard(W_u_rb, W_u);
    splitCheckerboard(W_v_rb, W_v);
    W_u_rb.red.copyTo(tempW_u.red);
    W_u_rb.black.copyTo(tempW_u.black);
    W_v_rb.red.copyTo(tempW_v.red);
    W_v_rb.black.copyTo(tempW_v.black);
    dW_u.red.setTo(0.0f);
    dW_u.black.setTo(0.0f);
    dW_v.red.setTo(0.0f);
    dW_v.black.setTo(0.0f);

    for (int i = 0; i < fixedPointIterations; i++)
    {
        CV_TRACE_REGION("fixedPoint_iteration");

        parallel_for_(Range(0, num_stripes), ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeDataTerm_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));

        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(
                                               *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, tempW_u, tempW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeSmoothnessTermHorPass_ParBody(
                                               *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, tempW_u, tempW_v, false));

        parallel_for_(Range(0, num_stripes),
                      ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, true));
        parallel_for_(Range(0, num_stripes),
                      ComputeSmoothnessTermVertPass_ParBody(*this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, false));

        for (int j = 0; j < sorIterations; j++)
        {
            CV_TRACE_REGION("SOR_iteration");
            parallel_for_(Range(0, num_stripes), RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
            parallel_for_(Range(0, num_stripes), RedBlackSOR_ParBody(*this, num_stripes, I0Mat.rows, dW_u, dW_v, false));
        }

        tempW_u.red = W_u_rb.red + dW_u.red;
        tempW_u.black = W_u_rb.black + dW_u.black;
        updateRepeatedBorders(tempW_u);
        tempW_v.red = W_v_rb.red + dW_v.red;
        tempW_v.black = W_v_rb.black + dW_v.black;
        updateRepeatedBorders(tempW_v);
    }
    mergeCheckerboard(W_u, tempW_u);
    mergeCheckerboard(W_v, tempW_v);
}

void AMVMaskedVariationalRefinementImpl::calcMasked(InputArray I0, InputArray I1, InputArray source_membership,
                                                     InputArray target_membership, InputOutputArray flow,
                                                     float active_threshold, float target_invalid_weight)
{
    CV_INSTRUMENT_REGION();

    const auto totalStarted = AMVProfileClock::now();
    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) ||
              (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!source_membership.empty() && source_membership.type() == CV_32FC1);
    CV_Assert(!target_membership.empty() && target_membership.type() == CV_32FC1);
    CV_Assert(I0.sameSize(source_membership));
    CV_Assert(I0.sameSize(target_membership));
    CV_Assert(!flow.empty() && flow.depth() == CV_32F && flow.channels() == 2);
    CV_Assert(I0.sameSize(flow));
    CV_Assert(active_threshold >= 0.0f && active_threshold <= 1.0f);
    CV_Assert(target_invalid_weight >= 0.0f && target_invalid_weight <= 1.0f);

    Mat uv[2];
    Mat &flowMat = flow.getMatRef();
    const auto splitStarted = AMVProfileClock::now();
    split(flowMat, uv);
    const double splitMs = elapsedMilliseconds(splitStarted);

    calcUVMasked(I0, I1, source_membership, target_membership, uv[0], uv[1], active_threshold,
                 target_invalid_weight);

    const auto mergeStarted = AMVProfileClock::now();
    merge(uv, 2, flowMat);
    const double mergeMs = elapsedMilliseconds(mergeStarted);

    lastMaskedProfile.wrapper_split_ms = splitMs;
    lastMaskedProfile.wrapper_merge_ms = mergeMs;
    lastMaskedProfile.used_uv_interface = 0;
    lastMaskedProfile.total_ms = elapsedMilliseconds(totalStarted);
}

void AMVMaskedVariationalRefinementImpl::calcUVMasked(InputArray I0, InputArray I1, InputArray source_membership,
                                                       InputArray target_membership, InputOutputArray flow_u,
                                                       InputOutputArray flow_v, float active_threshold,
                                                       float target_invalid_weight)
{
    CV_INSTRUMENT_REGION();

    lastMaskedProfile = AMVMaskedVRProfile();
    lastMaskedProfile.fixed_point_iterations = fixedPointIterations;
    lastMaskedProfile.sor_iterations_per_fixed_point = sorIterations;
    lastMaskedProfile.used_uv_interface = 1;
    const auto totalStarted = AMVProfileClock::now();

    CV_Assert(!I0.empty() && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert((I0.depth() == CV_8U && I1.depth() == CV_8U) ||
              (I0.depth() == CV_32F && I1.depth() == CV_32F));
    CV_Assert(!source_membership.empty() && source_membership.type() == CV_32FC1);
    CV_Assert(!target_membership.empty() && target_membership.type() == CV_32FC1);
    CV_Assert(I0.sameSize(source_membership));
    CV_Assert(I0.sameSize(target_membership));
    CV_Assert(!flow_u.empty() && flow_u.type() == CV_32FC1);
    CV_Assert(!flow_v.empty() && flow_v.type() == CV_32FC1);
    CV_Assert(I0.sameSize(flow_u));
    CV_Assert(flow_u.sameSize(flow_v));
    CV_Assert(active_threshold >= 0.0f && active_threshold <= 1.0f);
    CV_Assert(target_invalid_weight >= 0.0f && target_invalid_weight <= 1.0f);

    int num_stripes = getNumThreads();
    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    Mat sourceMat = source_membership.getMat();
    Mat targetMat = target_membership.getMat();
    Mat &W_u = flow_u.getMatRef();
    Mat &W_v = flow_v.getMatRef();

    prepareBuffers(I0Mat, I1Mat, W_u, W_v);
    prepareMaskedBuffers(sourceMat, targetMat, W_u, W_v, active_threshold, target_invalid_weight);

    const auto initializationStarted = AMVProfileClock::now();
    splitCheckerboard(W_u_rb, W_u);
    splitCheckerboard(W_v_rb, W_v);
    W_u_rb.red.copyTo(tempW_u.red);
    W_u_rb.black.copyTo(tempW_u.black);
    W_v_rb.red.copyTo(tempW_v.red);
    W_v_rb.black.copyTo(tempW_v.black);
    dW_u.red.setTo(0.0f);
    dW_u.black.setTo(0.0f);
    dW_v.red.setTo(0.0f);
    dW_v.black.setTo(0.0f);
    lastMaskedProfile.initialization_ms += elapsedMilliseconds(initializationStarted);

    for (int i = 0; i < fixedPointIterations; i++)
    {
        CV_TRACE_REGION("masked_fixedPoint_iteration");

        const auto gateStarted = AMVProfileClock::now();
        updateMaskedDataGate();
        lastMaskedProfile.data_gate_ms += elapsedMilliseconds(gateStarted);

        const auto dataStarted = AMVProfileClock::now();
        parallel_for_(Range(0, num_stripes), ComputeMaskedDataTerm_ParBody(
                                                   *this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeMaskedDataTerm_ParBody(
                                                   *this, num_stripes, I0Mat.rows, dW_u, dW_v, false));
        lastMaskedProfile.data_term_ms += elapsedMilliseconds(dataStarted);

        const auto smoothHorizontalStarted = AMVProfileClock::now();
        parallel_for_(Range(0, num_stripes), ComputeMaskedSmoothnessTermHorPass_ParBody(
                                                   *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb,
                                                   tempW_u, tempW_v, true));
        parallel_for_(Range(0, num_stripes), ComputeMaskedSmoothnessTermHorPass_ParBody(
                                                   *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb,
                                                   tempW_u, tempW_v, false));
        lastMaskedProfile.smoothness_horizontal_ms += elapsedMilliseconds(smoothHorizontalStarted);

        const auto smoothVerticalStarted = AMVProfileClock::now();
        parallel_for_(Range(0, num_stripes), ComputeMaskedSmoothnessTermVertPass_ParBody(
                                                   *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, true));
        parallel_for_(Range(0, num_stripes), ComputeMaskedSmoothnessTermVertPass_ParBody(
                                                   *this, num_stripes, I0Mat.rows, W_u_rb, W_v_rb, false));
        lastMaskedProfile.smoothness_vertical_ms += elapsedMilliseconds(smoothVerticalStarted);

        const auto sorStarted = AMVProfileClock::now();
        for (int j = 0; j < sorIterations; j++)
        {
            CV_TRACE_REGION("masked_SOR_iteration");
            parallel_for_(Range(0, num_stripes), MaskedRedBlackSOR_ParBody(
                                                       *this, num_stripes, I0Mat.rows, dW_u, dW_v, true));
            parallel_for_(Range(0, num_stripes), MaskedRedBlackSOR_ParBody(
                                                       *this, num_stripes, I0Mat.rows, dW_u, dW_v, false));
        }
        lastMaskedProfile.sor_ms += elapsedMilliseconds(sorStarted);

        const auto updateStarted = AMVProfileClock::now();
        tempW_u.red = W_u_rb.red + dW_u.red;
        tempW_u.black = W_u_rb.black + dW_u.black;
        updateRepeatedBorders(tempW_u);
        tempW_v.red = W_v_rb.red + dW_v.red;
        tempW_v.black = W_v_rb.black + dW_v.black;
        updateRepeatedBorders(tempW_v);
        lastMaskedProfile.flow_update_ms += elapsedMilliseconds(updateStarted);
    }

    const auto finalizeStarted = AMVProfileClock::now();
    mergeCheckerboard(W_u, tempW_u);
    mergeCheckerboard(W_v, tempW_v);
    W_u.setTo(0.0f, activeMask8 == 0);
    W_v.setTo(0.0f, activeMask8 == 0);
    lastMaskedProfile.finalize_ms += elapsedMilliseconds(finalizeStarted);
    lastMaskedProfile.total_ms = elapsedMilliseconds(totalStarted);
}

void AMVMaskedVariationalRefinementImpl::collectGarbage()
{
    CV_INSTRUMENT_REGION();

    Ix.release();
    Iy.release();
    Iz.release();
    Ixx.release();
    Ixy.release();
    Iyy.release();
    Ixz.release();
    Iyz.release();

    Ix_rb.release();
    Iy_rb.release();
    Iz_rb.release();
    Ixx_rb.release();
    Ixy_rb.release();
    Iyy_rb.release();
    Ixz_rb.release();
    Iyz_rb.release();

    A11.release();
    A12.release();
    A22.release();
    b1.release();
    b2.release();
    weights.release();

    sourceMembership_rb.release();
    activeMask_rb.release();
    dataGate_rb.release();
    weightsHor.release();
    weightsVert.release();
    sourceMembership.release();
    targetMembership.release();
    warpedTargetMembership.release();
    dataGate.release();
    currentW_u.release();
    currentW_v.release();
    activeMaskFloat.release();
    activeMask8.release();
    activeBeginRed.clear();
    activeEndRed.clear();
    activeBeginBlack.clear();
    activeEndBlack.clear();

    mapX.release();
    mapY.release();

    tempW_u.release();
    tempW_v.release();
    dW_u.release();
    dW_v.release();
    W_u_rb.release();
    W_v_rb.release();
}

Ptr<AMVMaskedVariationalRefinement> createAMVMaskedVariationalRefinement()
{ return makePtr<AMVMaskedVariationalRefinementImpl>(); }

}
