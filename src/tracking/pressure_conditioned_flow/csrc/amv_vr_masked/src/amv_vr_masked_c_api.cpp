#include "precomp.hpp"
#include "amv_vr_masked_factory.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define AMV_MASKED_EXPORT __declspec(dllexport)
#else
#define AMV_MASKED_EXPORT __attribute__((visibility("default")))
#endif

namespace
{

thread_local std::string g_last_error;

struct SolverHandle
{
    cv::Ptr<cv::AMVMaskedVariationalRefinement> solver;
};

struct AMVMaskedVRProfileC
{
    double total_ms;
    double wrapper_split_ms;
    double wrapper_merge_ms;
    double prepare_buffers_ms;
    double prepare_allocations_ms;
    double prepare_i1_conversion_ms;
    double prepare_warp_ms;
    double prepare_average_subtract_ms;
    double prepare_derivatives_ms;
    double prepare_masked_buffers_ms;
    double initialization_ms;
    double data_gate_ms;
    double data_term_ms;
    double smoothness_horizontal_ms;
    double smoothness_vertical_ms;
    double sor_ms;
    double flow_update_ms;
    double finalize_ms;
    int fixed_point_iterations;
    int sor_iterations_per_fixed_point;
    int used_uv_interface;
};

void set_error(const std::string &message)
{
    g_last_error = message;
}

bool finite_parameter(float value)
{
    return std::isfinite(static_cast<double>(value));
}

void validate_solver_parameters(
    float alpha,
    float delta,
    float gamma,
    int fixed_point_iterations,
    int sor_iterations,
    float omega)
{
    if (!finite_parameter(alpha) || alpha < 0.0f)
        throw std::invalid_argument("alpha must be finite and non-negative");
    if (!finite_parameter(delta) || delta < 0.0f)
        throw std::invalid_argument("delta must be finite and non-negative");
    if (!finite_parameter(gamma) || gamma < 0.0f)
        throw std::invalid_argument("gamma must be finite and non-negative");
    if (fixed_point_iterations <= 0)
        throw std::invalid_argument(
            "fixed_point_iterations must be positive");
    if (sor_iterations <= 0)
        throw std::invalid_argument(
            "sor_iterations must be positive");
    if (!finite_parameter(omega) || omega <= 0.0f || omega >= 2.0f)
        throw std::invalid_argument(
            "omega must be finite and in (0, 2)");
}

} // namespace

extern "C"
{

AMV_MASKED_EXPORT const char *amv_vr_masked_last_error()
{
    return g_last_error.c_str();
}

AMV_MASKED_EXPORT const char *amv_vr_masked_opencv_version()
{
    return CV_VERSION;
}

AMV_MASKED_EXPORT const char *amv_vr_masked_identity()
{
    return
        "OpenCV-4.13.0 exact VariationalRefinement fork; "
        "pressure-conditioned source/target membership; "
        "direct u/v in-place API; sparse source-active SOR; "
        "native phase profiling; no pressure diffusion";
}

AMV_MASKED_EXPORT void amv_vr_masked_set_num_threads(int value)
{
    cv::setNumThreads(value);
}

AMV_MASKED_EXPORT int amv_vr_masked_get_num_threads()
{
    return cv::getNumThreads();
}

AMV_MASKED_EXPORT int amv_vr_masked_set_sparse_sor_enabled(
    void *opaque_handle, int enabled)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument("solver handle is null");
        SolverHandle *handle = static_cast<SolverHandle *>(opaque_handle);
        handle->solver->setSparseMaskedSorEnabled(enabled != 0);
        return 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error("Unknown error while setting sparse SOR mode");
    }
    return -1;
}

AMV_MASKED_EXPORT int amv_vr_masked_get_sparse_sor_enabled(
    void *opaque_handle)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument("solver handle is null");
        SolverHandle *handle = static_cast<SolverHandle *>(opaque_handle);
        return handle->solver->getSparseMaskedSorEnabled() ? 1 : 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error("Unknown error while reading sparse SOR mode");
    }
    return -1;
}

AMV_MASKED_EXPORT void *amv_vr_masked_create(
    float alpha,
    float delta,
    float gamma,
    int fixed_point_iterations,
    int sor_iterations,
    float omega)
{
    g_last_error.clear();
    try
    {
        validate_solver_parameters(
            alpha,
            delta,
            gamma,
            fixed_point_iterations,
            sor_iterations,
            omega);

        std::unique_ptr<SolverHandle> handle(new SolverHandle());
        handle->solver =
            cv::createAMVMaskedVariationalRefinement();

        if (handle->solver.empty())
            throw std::runtime_error(
                "createAMVMaskedVariationalRefinement() "
                "returned null");

        handle->solver->setAlpha(alpha);
        handle->solver->setDelta(delta);
        handle->solver->setGamma(gamma);
        handle->solver->setFixedPointIterations(
            fixed_point_iterations);
        handle->solver->setSorIterations(sor_iterations);
        handle->solver->setOmega(omega);

        return handle.release();
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error(
            "Unknown error while creating masked AMV VR");
    }
    return nullptr;
}

AMV_MASKED_EXPORT void amv_vr_masked_destroy(
    void *opaque_handle)
{
    SolverHandle *handle =
        static_cast<SolverHandle *>(opaque_handle);
    delete handle;
}

AMV_MASKED_EXPORT int amv_vr_masked_refine_unmasked_f32(
    void *opaque_handle,
    const float *image0,
    const float *image1,
    int height,
    int width,
    float *flow_xy)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument(
                "solver handle is null");
        if (
            image0 == nullptr
            || image1 == nullptr
            || flow_xy == nullptr)
        {
            throw std::invalid_argument(
                "image or flow pointer is null");
        }
        if (height <= 1 || width <= 1)
            throw std::invalid_argument(
                "height and width must both exceed 1");

        SolverHandle *handle =
            static_cast<SolverHandle *>(opaque_handle);

        cv::Mat first(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(image0),
            static_cast<std::size_t>(width)
                * sizeof(float));
        cv::Mat second(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(image1),
            static_cast<std::size_t>(width)
                * sizeof(float));
        cv::Mat flow(
            height,
            width,
            CV_32FC2,
            flow_xy,
            static_cast<std::size_t>(width)
                * 2U * sizeof(float));

        handle->solver->calc(first, second, flow);
        return 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error(
            "Unknown error in unmasked AMV VR path");
    }
    return -1;
}

AMV_MASKED_EXPORT int amv_vr_masked_refine_f32(
    void *opaque_handle,
    const float *image0,
    const float *image1,
    const float *source_membership,
    const float *target_membership,
    int height,
    int width,
    float active_threshold,
    float target_invalid_weight,
    float *flow_xy)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument(
                "solver handle is null");
        if (
            image0 == nullptr
            || image1 == nullptr
            || source_membership == nullptr
            || target_membership == nullptr
            || flow_xy == nullptr)
        {
            throw std::invalid_argument(
                "image, membership, or flow pointer is null");
        }
        if (height <= 1 || width <= 1)
            throw std::invalid_argument(
                "height and width must both exceed 1");
        if (
            !finite_parameter(active_threshold)
            || active_threshold < 0.0f
            || active_threshold > 1.0f)
        {
            throw std::invalid_argument(
                "active_threshold must be in [0, 1]");
        }
        if (
            !finite_parameter(target_invalid_weight)
            || target_invalid_weight < 0.0f
            || target_invalid_weight > 1.0f)
        {
            throw std::invalid_argument(
                "target_invalid_weight must be in [0, 1]");
        }

        SolverHandle *handle =
            static_cast<SolverHandle *>(opaque_handle);

        const std::size_t image_step =
            static_cast<std::size_t>(width)
            * sizeof(float);
        const std::size_t flow_step =
            static_cast<std::size_t>(width)
            * 2U * sizeof(float);

        cv::Mat first(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(image0),
            image_step);
        cv::Mat second(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(image1),
            image_step);
        cv::Mat source(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(source_membership),
            image_step);
        cv::Mat target(
            height,
            width,
            CV_32FC1,
            const_cast<float *>(target_membership),
            image_step);
        cv::Mat flow(
            height,
            width,
            CV_32FC2,
            flow_xy,
            flow_step);

        handle->solver->calcMasked(
            first,
            second,
            source,
            target,
            flow,
            active_threshold,
            target_invalid_weight);
        return 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error(
            "Unknown error in masked AMV VR path");
    }
    return -1;
}


AMV_MASKED_EXPORT int amv_vr_masked_refine_uv_f32(
    void *opaque_handle,
    const float *image0,
    const float *image1,
    const float *source_membership,
    const float *target_membership,
    int height,
    int width,
    float active_threshold,
    float target_invalid_weight,
    float *flow_u,
    float *flow_v)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument("solver handle is null");
        if (
            image0 == nullptr
            || image1 == nullptr
            || source_membership == nullptr
            || target_membership == nullptr
            || flow_u == nullptr
            || flow_v == nullptr)
        {
            throw std::invalid_argument(
                "image, membership, or u/v flow pointer is null");
        }
        if (height <= 1 || width <= 1)
            throw std::invalid_argument(
                "height and width must both exceed 1");
        if (
            !finite_parameter(active_threshold)
            || active_threshold < 0.0f
            || active_threshold > 1.0f)
        {
            throw std::invalid_argument(
                "active_threshold must be in [0, 1]");
        }
        if (
            !finite_parameter(target_invalid_weight)
            || target_invalid_weight < 0.0f
            || target_invalid_weight > 1.0f)
        {
            throw std::invalid_argument(
                "target_invalid_weight must be in [0, 1]");
        }

        SolverHandle *handle =
            static_cast<SolverHandle *>(opaque_handle);
        const std::size_t image_step =
            static_cast<std::size_t>(width) * sizeof(float);

        cv::Mat first(
            height, width, CV_32FC1,
            const_cast<float *>(image0), image_step);
        cv::Mat second(
            height, width, CV_32FC1,
            const_cast<float *>(image1), image_step);
        cv::Mat source(
            height, width, CV_32FC1,
            const_cast<float *>(source_membership), image_step);
        cv::Mat target(
            height, width, CV_32FC1,
            const_cast<float *>(target_membership), image_step);
        cv::Mat u(height, width, CV_32FC1, flow_u, image_step);
        cv::Mat v(height, width, CV_32FC1, flow_v, image_step);

        handle->solver->calcUVMasked(
            first, second, source, target, u, v,
            active_threshold, target_invalid_weight);
        return 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error("Unknown error in masked AMV VR u/v path");
    }
    return -1;
}

AMV_MASKED_EXPORT int amv_vr_masked_profile_abi_version()
{
    return 1;
}

AMV_MASKED_EXPORT int amv_vr_masked_get_last_profile(
    void *opaque_handle,
    AMVMaskedVRProfileC *output)
{
    g_last_error.clear();
    try
    {
        if (opaque_handle == nullptr)
            throw std::invalid_argument("solver handle is null");
        if (output == nullptr)
            throw std::invalid_argument("profile output pointer is null");

        SolverHandle *handle =
            static_cast<SolverHandle *>(opaque_handle);
        const cv::AMVMaskedVRProfile p =
            handle->solver->getLastMaskedProfile();

        output->total_ms = p.total_ms;
        output->wrapper_split_ms = p.wrapper_split_ms;
        output->wrapper_merge_ms = p.wrapper_merge_ms;
        output->prepare_buffers_ms = p.prepare_buffers_ms;
        output->prepare_allocations_ms = p.prepare_allocations_ms;
        output->prepare_i1_conversion_ms = p.prepare_i1_conversion_ms;
        output->prepare_warp_ms = p.prepare_warp_ms;
        output->prepare_average_subtract_ms = p.prepare_average_subtract_ms;
        output->prepare_derivatives_ms = p.prepare_derivatives_ms;
        output->prepare_masked_buffers_ms = p.prepare_masked_buffers_ms;
        output->initialization_ms = p.initialization_ms;
        output->data_gate_ms = p.data_gate_ms;
        output->data_term_ms = p.data_term_ms;
        output->smoothness_horizontal_ms = p.smoothness_horizontal_ms;
        output->smoothness_vertical_ms = p.smoothness_vertical_ms;
        output->sor_ms = p.sor_ms;
        output->flow_update_ms = p.flow_update_ms;
        output->finalize_ms = p.finalize_ms;
        output->fixed_point_iterations = p.fixed_point_iterations;
        output->sor_iterations_per_fixed_point =
            p.sor_iterations_per_fixed_point;
        output->used_uv_interface = p.used_uv_interface;
        return 0;
    }
    catch (const cv::Exception &exc)
    {
        set_error(std::string("OpenCV error: ") + exc.what());
    }
    catch (const std::exception &exc)
    {
        set_error(exc.what());
    }
    catch (...)
    {
        set_error("Unknown error while reading masked AMV VR profile");
    }
    return -1;
}

AMV_MASKED_EXPORT void amv_vr_masked_collect_garbage(
    void *opaque_handle)
{
    if (opaque_handle == nullptr)
        return;

    SolverHandle *handle =
        static_cast<SolverHandle *>(opaque_handle);
    handle->solver->collectGarbage();
}

} // extern "C"
