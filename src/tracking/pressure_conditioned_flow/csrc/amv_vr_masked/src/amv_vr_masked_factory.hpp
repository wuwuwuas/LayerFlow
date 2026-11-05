#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace cv
{

struct AMVMaskedVRProfile
{
    double total_ms = 0.0;
    double wrapper_split_ms = 0.0;
    double wrapper_merge_ms = 0.0;
    double prepare_buffers_ms = 0.0;
    double prepare_allocations_ms = 0.0;
    double prepare_i1_conversion_ms = 0.0;
    double prepare_warp_ms = 0.0;
    double prepare_average_subtract_ms = 0.0;
    double prepare_derivatives_ms = 0.0;
    double prepare_masked_buffers_ms = 0.0;
    double initialization_ms = 0.0;
    double data_gate_ms = 0.0;
    double data_term_ms = 0.0;
    double smoothness_horizontal_ms = 0.0;
    double smoothness_vertical_ms = 0.0;
    double sor_ms = 0.0;
    double flow_update_ms = 0.0;
    double finalize_ms = 0.0;
    int fixed_point_iterations = 0;
    int sor_iterations_per_fixed_point = 0;
    int used_uv_interface = 0;
};

class AMVMaskedVariationalRefinement : public VariationalRefinement
{
public:
    virtual void calcMasked(
        InputArray I0,
        InputArray I1,
        InputArray source_membership,
        InputArray target_membership,
        InputOutputArray flow,
        float active_threshold,
        float target_invalid_weight
    ) = 0;

    virtual void calcUVMasked(
        InputArray I0,
        InputArray I1,
        InputArray source_membership,
        InputArray target_membership,
        InputOutputArray flow_u,
        InputOutputArray flow_v,
        float active_threshold,
        float target_invalid_weight
    ) = 0;

    virtual AMVMaskedVRProfile getLastMaskedProfile() const = 0;
    virtual bool getSparseMaskedSorEnabled() const = 0;
    virtual void setSparseMaskedSorEnabled(bool val) = 0;
};

Ptr<AMVMaskedVariationalRefinement>
createAMVMaskedVariationalRefinement();

} // namespace cv
