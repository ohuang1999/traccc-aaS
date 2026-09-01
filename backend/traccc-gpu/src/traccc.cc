// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cuda_runtime_api.h>

#include <chrono>

#include "TracccGpuStandalone.hpp"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

namespace triton { namespace backend { namespace Traccc {

//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
    // Delete the "global" state associated with the backend.
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
    std::string* state = reinterpret_cast<std::string*>(vstate);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
            .c_str());

    delete state;

    return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//! Possible differences in this class!
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

    // Name of the input and output tensor
    const std::string &InputCellPositionsTensorName() const { return input_cell_positions_name_; }
    const std::string &InputCellPropertiesTensorName() const { return input_cell_properties_name_; }
    const std::string &OutputTensorName() const { return output_name_; }

    // Datatype of the input and output tensor
    TRITONSERVER_DataType InputCellPositionsTensorDataType() const { return input_cell_positions_datatype_; }
    TRITONSERVER_DataType InputCellPropertiesTensorDataType() const { return input_cell_properties_datatype_; }
    TRITONSERVER_DataType OutputTensorDataType() const
    {
        return output_datatype_;
    }

    // Shape of the input and output tensor as given in the model
    // configuration file. This shape will not include the batch
    // dimension (if the model has one).
    // const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_;
    // }

    // Shape of the input and output tensor, including the batch
    // dimension (if the model has one). This method cannot be called
    // until the model is completely loaded and initialized, including
    // all instances of the model. In practice, this means that backend
    // should only call it in TRITONBACKEND_ModelInstanceExecute.
    const std::vector<int64_t> &InputCellPositionsTensorNonBatchShape() const
    {
        return input_cell_positions_nb_shape_;
    }
    const std::vector<int64_t> &InputCellPropertiesTensorNonBatchShape() const
    {
        return input_cell_properties_nb_shape_;
    }
    const std::vector<int64_t> &OutputTensorNonBatchShape() const
    {
        return output_nb_shape_;
    }

    // Validate that this model is supported by this backend.
    TRITONSERVER_Error *ValidateModelConfig();

    //   std::string model_path;
    int64_t cellFeatures;

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

    std::string input_cell_positions_name_;
    std::string input_cell_properties_name_;
    std::string output_name_;

    TRITONSERVER_DataType input_cell_positions_datatype_;
    TRITONSERVER_DataType input_cell_properties_datatype_;
    TRITONSERVER_DataType output_datatype_;

    std::vector<int64_t> input_cell_positions_nb_shape_;
    std::vector<int64_t> input_cell_properties_nb_shape_;
    std::vector<int64_t> input_cell_positions_shape_;
    std::vector<int64_t> input_cell_properties_shape_;
    std::vector<int64_t> output_nb_shape_;
    std::vector<int64_t> output_shape_;

    bool shape_initialized_;
};

//! Also possible problems in this function!
ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
    // If verbose logging is enabled, dump the model's configuration as
    // JSON into the console output.
    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
    }

    // ModelConfig is the model configuration as a TritonJson
    // object. Use the TritonJson utilities to parse the JSON and
    // determine if the configuration is supported by this backend.
    common::TritonJson::Value inputs, outputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

    // The model must have exactly 2 inputs and 3 output.
    RETURN_ERROR_IF_FALSE(
        inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 2 inputs"));
    RETURN_ERROR_IF_FALSE(
        outputs.ArraySize() == 4, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model configuration must have 4 outputs"));

    common::TritonJson::Value input_cell_positions, input_cell_properties, output;
    RETURN_IF_ERROR(inputs.IndexAsObject(0, &input_cell_positions));
    RETURN_IF_ERROR(inputs.IndexAsObject(1, &input_cell_properties));
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

    // Record the input and output name in the model state.
    const char *input_cell_positions_name, *input_cell_properties_name;
    size_t input_cell_positions_len, input_cell_properties_len;
    RETURN_IF_ERROR(input_cell_positions.MemberAsString("name", &input_cell_positions_name, &input_cell_positions_len));
    RETURN_IF_ERROR(input_cell_properties.MemberAsString("name", &input_cell_properties_name, &input_cell_properties_len));
    input_cell_positions_name_ = std::string(input_cell_positions_name);
    input_cell_properties_name_ = std::string(input_cell_properties_name);

    const char *output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
    output_name_ = std::string(output_name);

    // Input and output must have same datatype
    std::string input_cell_positions_dtype, input_cell_properties_dtype, output_dtype;
    RETURN_IF_ERROR(input_cell_positions.MemberAsString("data_type", &input_cell_positions_dtype));
    RETURN_IF_ERROR(input_cell_properties.MemberAsString("data_type", &input_cell_properties_dtype));
    RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
    // RETURN_ERROR_IF_FALSE(
    //     input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
    //     std::string("expected input and output datatype to match, got ") +
    //         input_dtype + " and " + output_dtype);
    input_cell_positions_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_cell_positions_dtype);
    input_cell_properties_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_cell_properties_dtype);
    output_datatype_ = ModelConfigDataTypeToTritonServerDataType(output_dtype);

    std::vector<int64_t> input_cell_positions_shape, input_cell_properties_shape, output_shape;
    RETURN_IF_ERROR(backend::ParseShape(input_cell_positions, "dims", &input_cell_positions_shape));
    RETURN_IF_ERROR(backend::ParseShape(input_cell_properties, "dims", &input_cell_properties_shape));
    RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

    input_cell_positions_nb_shape_ = input_cell_positions_shape;
    input_cell_properties_nb_shape_ = input_cell_properties_shape;
    output_nb_shape_ = output_shape;

    return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance 
{
private:

    ModelState* model_state_;

    ModelInstanceState(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance)
        : BackendModelInstance(model_state, triton_model_instance),
            model_state_(model_state),
            host_mr_(),
            device_mr_(DeviceId())
    {
    }

public:
    static TRITONSERVER_Error* Create(
        ModelState* model_state,
        TRITONBACKEND_ModelInstance* triton_model_instance,
        ModelInstanceState** state);
    virtual ~ModelInstanceState() = default;

    // Get the state of the model that corresponds to this instance.
    ModelState* StateForModel() const { return model_state_; }

    // define standalone object
    std::unique_ptr<TracccGpuStandalone> traccc_gpu_standalone_;

    // Memory resources
    vecmem::host_memory_resource host_mr_;
    vecmem::cuda::device_memory_resource device_mr_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
    // Get the model state associated with this instance's model.
    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void* vmodelstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

    // Create a ModelInstanceState object and associate it with the
    // TRITONBACKEND_ModelInstance.
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(
        ModelInstanceState::Create(model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    // Set the CUDA device for this thread
    cudaError_t err = cudaSetDevice(instance_state->DeviceId());
    if (err != cudaSuccess)
    {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to set CUDA device: " + std::string(cudaGetErrorString(err))).c_str());
    }
  
    // Geometry loading throws on failure, and this is a C API boundary: an
    // escaping exception calls std::terminate and takes the whole tritonserver
    // down with it. Turn it into a failed model load instead, so the server
    // survives and the reason reaches the log.
    try
    {
        instance_state->traccc_gpu_standalone_ = std::make_unique<TracccGpuStandalone>(
            &instance_state->host_mr_,
            &instance_state->device_mr_,
            instance_state->DeviceId()
        );
    }
    catch (const std::exception& e)
    {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to initialize traccc: " + std::string(e.what())).c_str());
    }
    return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    // Collect various timestamps during the execution of this batch or
    // requests. These values are reported below before returning from
    // the function.

    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    // Triton will not call this function simultaneously for the same
    // 'instance'. But since this backend could be used by multiple
    // instances from multiple models the implementation needs to handle
    // multiple calls to this function at the same time (with different
    // 'instance' objects). Best practice for a high-performance
    // implementation is to avoid introducing mutex/lock and instead use
    // only function-local and model-instance-specific state.
    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&instance_state)));
    ModelState* model_state = instance_state->StateForModel();

    // Set the CUDA device for this thread
    // Seems that this is necessary to set the device for each request
    // Without leads to out-of-bounds memory access error
    cudaError_t err = cudaSetDevice(instance_state->DeviceId());
    if (err != cudaSuccess)
    {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to set CUDA device: " + std::string(cudaGetErrorString(err))).c_str());
    }

    // 'responses' is initialized as a parallel array to 'requests',
    // with one TRITONBACKEND_Response object for each
    // TRITONBACKEND_Request object. If something goes wrong while
    // creating these response objects, the backend simply returns an
    // error from TRITONBACKEND_ModelInstanceExecute, indicating to
    // Triton that this backend did not create or send any responses and
    // so it is up to Triton to create and send an appropriate error
    // response for each request. RETURN_IF_ERROR is one of several
    // useful macros for error handling that can be found in
    // backend_common.h.

    std::vector<TRITONBACKEND_Response*> responses;
    responses.reserve(request_count);
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
        responses.push_back(response);
    }

    // At this point, the backend takes ownership of 'requests', which
    // means that it is responsible for sending a response for every
    // request. From here, even if something goes wrong in processing,
    // the backend must return 'nullptr' from this function to indicate
    // success. Any errors and failures must be communicated via the
    // response objects.
    //
    // To simplify error handling, the backend utilities manage
    // 'responses' in a specific way and it is recommended that backends
    // follow this same pattern. When an error is detected in the
    // processing of a request, an appropriate error response is sent
    // and the corresponding TRITONBACKEND_Response object within
    // 'responses' is set to nullptr to indicate that the
    // request/response has already been handled and no further processing
    // should be performed for that request. Even if all responses fail,
    // the backend still allows execution to flow to the end of the
    // function so that statistics are correctly reported by the calls
    // to TRITONBACKEND_ModelInstanceReportStatistics and
    // TRITONBACKEND_ModelInstanceReportBatchStatistics.
    // RESPOND_AND_SET_NULL_IF_ERROR, and
    // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
    // backend_common.h that assist in this management of response
    // objects.

    // The backend could iterate over the 'requests' and process each
    // one separately. But for performance reasons it is usually
    // preferred to create batched input tensors that are processed
    // simultaneously. This is especially true for devices like GPUs
    // that are capable of exploiting the large amount parallelism
    // exposed by larger data sets.
    //
    // The backend utilities provide a "collector" to facilitate this
    // batching process. The 'collector's ProcessTensor function will
    // combine a tensor's value from each request in the batch into a
    // single contiguous buffer. The buffer can be provided by the
    // backend or 'collector' can create and manage it. In this backend,
    // there is not a specific buffer into which the batch should be
    // created, so use ProcessTensor arguments that cause collector to
    // manage it. ProcessTensor does NOT support TRITONSERVER_TYPE_BYTES
    // data type.

    BackendInputCollector collector(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        false /* pinned_enabled */, nullptr /* stream*/);

    // To instruct ProcessTensor to "gather" the entire batch of input
    // tensors into a single contiguous buffer in CPU memory, set the
    // "allowed input types" to be the CPU ones (see tritonserver.h in
    // the triton-inference-server/core repo for allowed memory types).
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
        {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};

    const char *input_cell_positions_buffer, *input_cell_properties_buffer;
    size_t input_cell_positions_buffer_byte_size, input_cell_properties_buffer_byte_size;
    TRITONSERVER_MemoryType input_cell_positions_buffer_memory_type, input_cell_properties_buffer_memory_type;
    int64_t input_cell_positions_buffer_memory_type_id, input_cell_properties_buffer_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            model_state->InputCellPositionsTensorName().c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_cell_positions_buffer,
            &input_cell_positions_buffer_byte_size, &input_cell_positions_buffer_memory_type,
            &input_cell_positions_buffer_memory_type_id));

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(
            model_state->InputCellPropertiesTensorName().c_str(), nullptr /* existing_buffer */,
            0 /* existing_buffer_byte_size */, allowed_input_types, &input_cell_properties_buffer,
            &input_cell_properties_buffer_byte_size, &input_cell_properties_buffer_memory_type,
            &input_cell_properties_buffer_memory_type_id));

    // Finalize the collector. If 'true' is returned, 'input_buffer'
    // will not be valid until the backend synchronizes the CUDA
    // stream or event that was used when creating the collector. For
    // this backend, GPU is not supported and so no CUDA sync should
    // be needed; so if 'true' is returned simply log an error.
    const bool need_cuda_input_sync = collector.Finalize();
    if (need_cuda_input_sync)
    {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            "'Traccc' backend: unexpected CUDA sync required by collector");
    }

    // 'input_buffer' contains the batched input tensor. The backend can
    // implement whatever logic is necessary to produce the output
    // tensor. This backend simply logs the input tensor value and then
    // returns the input tensor value in the output tensor so no actual
    // computation is needed.

    uint64_t compute_start_ns = 0;
    SET_TIMESTAMP(compute_start_ns);

    auto input_proc_start = std::chrono::high_resolution_clock::now();

    // Determine the number of objects in the input buffer
    size_t num_cells = input_cell_positions_buffer_byte_size / (sizeof(std::int64_t) * 4);
    size_t num_props = input_cell_properties_buffer_byte_size / (sizeof(float) * 2);

    // convert to expected types
    const std::int64_t *cell_positions_ptr = 
        reinterpret_cast<const std::int64_t *>(input_cell_positions_buffer);
    const float *cell_properties_ptr = 
        reinterpret_cast<const float *>(input_cell_properties_buffer);

    if (num_cells != num_props) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, 
            "Mismatch between number of cell positions and cell properties.");
    }

    std::cout << "Number of cells received: " << num_cells  << std::endl;
    // for (size_t i = 0; i < 5 && i < num_cells; ++i) {
    //     std::cout << "Cell " << i << ": ";
    //     std::cout << "pos=[";
    //     std::cout << cell_positions_ptr[i * 4] << ", ";
    //     std::cout << cell_positions_ptr[i * 4 + 1] << ", ";
    //     std::cout << cell_positions_ptr[i * 4 + 2] << ", ";
    //     std::cout << cell_positions_ptr[i * 4 + 3] << "], props=[";
    //     std::cout << cell_properties_ptr[i * 2] << ", ";
    //     std::cout << cell_properties_ptr[i * 2 + 1] << "]" << std::endl;
    // }

    // Read measurements into traccc 
    std::vector<traccc::io::csv::cell> cells = instance_state->traccc_gpu_standalone_->read_from_array(
        cell_positions_ptr, cell_properties_ptr, num_cells, true);
    
    auto input_proc_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] Input processing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(input_proc_end - input_proc_start).count()
              << " ms" << std::endl;

    // run the reco chain
    bool print_stats = false;
    auto traccc_result = instance_state->traccc_gpu_standalone_->run(cells, print_stats);

    auto output_proc_start = std::chrono::high_resolution_clock::now();

    uint64_t compute_end_ns = 0;
    SET_TIMESTAMP(compute_end_ns);

    bool supports_first_dim_batching;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

    // Because the output tensor values are concatenated into a single
    // contiguous 'output_buffer', the backend must "scatter" them out
    // to the individual response output tensors.  The backend utilities
    // provide a "responder" to facilitate this scattering process.
    // BackendOutputResponder does NOT support TRITONSERVER_TYPE_BYTES
    // data type.

    // Initialize the responder
    BackendOutputResponder responder(
        requests, request_count, &responses, model_state->TritonMemoryManager(),
        supports_first_dim_batching, false /* pinned_enabled */,
        nullptr /* stream*/);

    // The 'responders's ProcessTensor function will copy the portion of
    // 'output_buffer' corresponding to each request's output into the
    // response for that request.

    // Process the outputs
    {
        // --------------- Process 'TRK_PARAMS', 'MEASUREMENTS', and 'GEOMETRY_IDS' ---------------
        size_t num_tracks = traccc_result.tracks_and_states.tracks.size();
        
        // Buffers for the output tensors
        std::vector<float> trk_params_buffer;
        std::vector<float> measurements_buffer;
        std::vector<float> covariances_buffer;
        std::vector<int64_t> geometry_ids_buffer;

        trk_params_buffer.reserve(num_tracks * 8);
        measurements_buffer.reserve(num_tracks * 15 * 6); 
        covariances_buffer.reserve(num_tracks * 15 * 25);
        geometry_ids_buffer.reserve(num_tracks * 15);

        // Track exclusion counters
        int excluded_non_positive_ndf = 0;
        int excluded_not_all_smoothed = 0;
        int excluded_unknown = 0;
        int excluded_no_state = 0;
        int included_tracks = 0;

        // Process all tracks
        for (size_t i = 0; i < num_tracks; ++i) {
            const auto& track = traccc_result.tracks_and_states.tracks.at(i);

            // Check track fit outcome
            auto track_fit_outcome = track.fit_outcome();
            if (track_fit_outcome != traccc::track_fit_outcome::SUCCESS) {
                ++excluded_unknown;
                continue;
            }
            if (track.ndf() < 0) {
                excluded_non_positive_ndf += 1;
                continue;
            }
            if (track.constituent_links().size() < 3) {
                excluded_no_state += 1;
                continue;
            }

            // Add separator before this track's measurements, if it's not the first included track
            // This is done only for geometry ids, and splits on the track are then done on this
            // variable from the client side. 
            if (included_tracks > 0) {
                geometry_ids_buffer.push_back(0);
            }

            // --- Process Track Parameters ---
            trk_params_buffer.push_back(static_cast<float>(track.chi2()));
            trk_params_buffer.push_back(static_cast<float>(track.ndf()));

            const auto& fitted_params = track.params();
            trk_params_buffer.push_back(static_cast<float>(fitted_params.bound_local()[0]));
            trk_params_buffer.push_back(static_cast<float>(fitted_params.bound_local()[1]));
            trk_params_buffer.push_back(static_cast<float>(fitted_params.phi()));
            trk_params_buffer.push_back(static_cast<float>(fitted_params.theta()));
            trk_params_buffer.push_back(static_cast<float>(fitted_params.qop()));
            trk_params_buffer.push_back(static_cast<float>(fitted_params.time()));

            if (included_tracks < 3 && print_stats)
            {
                std::cout << "Track " << included_tracks << " parameters: ";
                std::cout << static_cast<float>(track.chi2()) << " ";
                std::cout << static_cast<float>(track.ndf()) << " ";
                std::cout << static_cast<float>(fitted_params.bound_local()[0]) << " ";
                std::cout << static_cast<float>(fitted_params.bound_local()[1]) << " ";
                std::cout << static_cast<float>(fitted_params.phi()) << " ";
                std::cout << static_cast<float>(fitted_params.theta()) << " ";
                std::cout << static_cast<float>(fitted_params.qop()) << " ";
                std::cout << static_cast<float>(fitted_params.time()) << std::endl;
            }

            // --- Process Measurements for this track ---
            const auto& constituent_links = track.constituent_links();
            for (size_t j = 0; j < constituent_links.size(); ++j) {
                const auto& link = constituent_links[j];
                
                if (link.type != traccc::edm::track_constituent_link::track_state) {
                    continue;
                }
                
                size_t state_idx = link.index;
                auto const& state = traccc_result.tracks_and_states.states.at(state_idx);
                auto const& measurement =
                    traccc_result.measurements.at(state.measurement_index());
                
                // Use the measurement local position and variance
                measurements_buffer.push_back(measurement.local_position()[0]); // local x
                measurements_buffer.push_back(measurement.local_position()[1]); // local y

                auto const& smoothed_params = state.smoothed_params();
                measurements_buffer.push_back(smoothed_params.phi());
                measurements_buffer.push_back(smoothed_params.theta());
                measurements_buffer.push_back(smoothed_params.qop());
                measurements_buffer.push_back(smoothed_params.time());

                auto const& cov = state.smoothed_params().covariance();
                // Covariance matrix (5x5) flattened in row-major order
                // TODO: only need to send upper triangle since symmetric
                for (size_t row = 0; row < 5; ++row) {
                    for (size_t col = 0; col < 5; ++col) {
                        // check for nan or inf
                        float value = static_cast<float>(cov[row][col]);
                        if (std::isnan(value) || std::isinf(value) || (value > 1e8)) {
                            covariances_buffer.push_back(0.0f); // fallback to 0.0f
                        } else {
                            covariances_buffer.push_back(value);
                        }
                    }
                }

                uint64_t detray_id = measurement.surface_link().value();
                try {
                    geometry_ids_buffer.push_back(
                        instance_state->traccc_gpu_standalone_->getDetrayToAthenaMap().at(detray_id));
                } catch (const std::out_of_range& e) {
                    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, 
                                ("Missing reverse mapping for Detray ID: " 
                                    + std::to_string(detray_id)).c_str());
                    geometry_ids_buffer.push_back(detray_id); // Fallback
                }
            }

            ++included_tracks;
        }

        // Log exclusion statistics
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("Track Exclusion Summary - Total: ") + std::to_string(num_tracks) +
                     ", Excluded (non-positive NDF): " + std::to_string(excluded_non_positive_ndf) +
                     ", Excluded (not all smoothed): " + std::to_string(excluded_not_all_smoothed) +
                     ", Excluded (unknown): " + std::to_string(excluded_unknown) +
                     ", Excluded (no state): " + std::to_string(excluded_no_state) +
                     ", Included: " + std::to_string(included_tracks)).c_str());

        // --- Send 'TRK_PARAMS' tensor ---
        std::vector<int64_t> trk_params_shape = {static_cast<int64_t>(included_tracks), 8};
        responder.ProcessTensor(
            "TRK_PARAMS", TRITONSERVER_TYPE_FP32, trk_params_shape,
            reinterpret_cast<const char*>(trk_params_buffer.data()),
            TRITONSERVER_MEMORY_CPU, 0);

        // --- Send 'MEASUREMENTS' tensor ---
        std::vector<int64_t> measurements_shape 
            = {static_cast<int64_t>(measurements_buffer.size() / 6), 6};
        responder.ProcessTensor(
            "MEASUREMENTS", TRITONSERVER_TYPE_FP32, measurements_shape,
            reinterpret_cast<const char*>(measurements_buffer.data()),
            TRITONSERVER_MEMORY_CPU, 0);

        // --- Send 'COVARIANCES' tensor ---
        std::vector<int64_t> covariances_shape 
            = {static_cast<int64_t>(covariances_buffer.size() / 25), 25};
        responder.ProcessTensor(
            "COVARIANCES", TRITONSERVER_TYPE_FP32, covariances_shape,
            reinterpret_cast<const char*>(covariances_buffer.data()),
            TRITONSERVER_MEMORY_CPU, 0);

        // --- Send 'GEOMETRY_IDS' tensor ---
        std::vector<int64_t> geometry_ids_shape = {static_cast<int64_t>(geometry_ids_buffer.size())};
        responder.ProcessTensor(
            "GEOMETRY_IDS", TRITONSERVER_TYPE_INT64, geometry_ids_shape,
            reinterpret_cast<const char*>(geometry_ids_buffer.data()),
            TRITONSERVER_MEMORY_CPU, 0);
    }

    auto output_proc_end = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMING] Output processing: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    output_proc_end - output_proc_start).count()
              << " ms" << std::endl;

    // Finalize the responder. If 'true' is returned, the output
    // tensors' data will not be valid until the backend synchronizes
    // the CUDA stream or event that was used when creating the
    // responder. For this backend, GPU is not supported and so no CUDA
    // sync should be needed; so if 'true' is returned simply log an
    // error.
    const bool need_cuda_output_sync = responder.Finalize();
    if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'traccc' backend: unexpected CUDA sync required by responder");
    }

    // Send all the responses that haven't already been sent because of
    // an earlier error.
    for (auto& response : responses) {
    if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send response");
    }
    }

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
    // For batch statistics need to know the total batch size of the
    // requests. This is not necessarily just the number of requests,
    // because if the model supports batching then any request can be a
    // batched request itself.
    size_t total_batch_size = 0;
    if (!supports_first_dim_batching) {
    total_batch_size = request_count;
    } else {
        for (uint32_t r = 0; r < request_count; ++r) 
        {
            auto& request = requests[r];
            TRITONBACKEND_Input* input = nullptr;
            LOG_IF_ERROR(
                TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
                "failed getting request input");
            if (input != nullptr) 
            {
                const int64_t* shape = nullptr;
                LOG_IF_ERROR(
                    TRITONBACKEND_InputProperties(
                        input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
                    "failed getting input properties");
                if (shape != nullptr) 
                {
                    total_batch_size += shape[0];
                }
            }
        }
    }
#else
    (void)exec_start_ns;
    (void)exec_end_ns;
    (void)compute_start_ns;
    (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

// Report statistics for each request, and then release the request.
for (uint32_t r = 0; r < request_count; ++r) 
{
    auto& request = requests[r];

    #ifdef TRITON_ENABLE_STATS
        LOG_IF_ERROR(
            TRITONBACKEND_ModelInstanceReportStatistics(
                instance_state->TritonModelInstance(), request,
                (responses[r] != nullptr) /* success */, exec_start_ns,
                compute_start_ns, compute_end_ns, exec_end_ns),
            "failed reporting request statistics");
    #endif  // TRITON_ENABLE_STATS

        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "failed releasing request");
}

#ifdef TRITON_ENABLE_STATS
    // Report batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            instance_state->TritonModelInstance(), total_batch_size,
            exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::traccc
