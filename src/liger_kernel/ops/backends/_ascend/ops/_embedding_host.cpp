// Host preparation and autograd only. All embedding math remains in Triton.
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <c10/util/SmallVector.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <string>
#include <unordered_map>

namespace py = pybind11;
using torch::Tensor;

namespace TORCH_EXTENSION_NAME {

struct Key {
  std::array<int64_t, 8> fields;
  std::string debug;
  bool operator==(const Key& other) const {
    return fields == other.fields && debug == other.debug;
  }
};
struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t hash = std::hash<std::string>{}(key.debug);
    for (auto field : key.fields)
      hash ^= std::hash<int64_t>{}(field) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};
struct Plan {
  py::object owner, kernel, launch, stream, prefix, count, device;
  bool raw_allowed;
};
struct State {
  py::object cold, backward, compiled_type, current_device, limit, runtime_utils;
  py::object enter_key, exit_key, pre_key, profiler_key;
  std::unordered_map<Key, Plan, KeyHash> plans;
  std::deque<Key> order;
};
// Process-wide callbacks and metadata; no user tensors or addresses are retained.
// Python teardown can outlive extension teardown, so do not decref at process exit.
static State* state = nullptr;

void clear_cache() {
  if (state) { state->plans.clear(); state->order.clear(); }
}
size_t cache_size() { return state ? state->plans.size() : 0; }

void configure(py::object cold, py::object backward, py::object compiled_type,
               py::object current_device, py::object limit, py::object runtime_utils) {
  if (!state) state = new State();
  clear_cache();
  state->cold = cold; state->backward = backward;
  state->compiled_type = compiled_type; state->current_device = current_device;
  state->limit = limit; state->runtime_utils = runtime_utils;
  state->enter_key = py::str("launch_enter_hook");
  state->exit_key = py::str("launch_exit_hook");
  state->pre_key = py::str("pre_run_hooks");
  state->profiler_key = py::str("TRITON_PROFILER_REGISTERED");
}

bool hooks_active(const Plan& plan) {
  auto enter = py::reinterpret_steal<py::object>(PyObject_GetAttr(state->compiled_type.ptr(), state->enter_key.ptr()));
  auto exit = py::reinterpret_steal<py::object>(PyObject_GetAttr(state->compiled_type.ptr(), state->exit_key.ptr()));
  auto pre = py::reinterpret_steal<py::object>(PyObject_GetAttr(plan.kernel.ptr(), state->pre_key.ptr()));
  if (!enter || !exit || !pre) throw py::error_already_set();
  const int pre_active = PyObject_IsTrue(pre.ptr());
  if (pre_active < 0) throw py::error_already_set();
  return !enter.is_none() || !exit.is_none() || pre_active;
}

Tensor cold_forward(const Tensor& weight, const Tensor& indices, const Key* key) {
  auto result = state->cold(weight, indices).cast<py::tuple>();
  auto output = result[0].cast<Tensor>();
  if (key && !result[1].is_none()) {
    auto source = result[1].cast<py::tuple>();
    auto runtime = source[4];
    bool allowed = !source[5].is_none() && !source[6].is_none()
        && py::hasattr(runtime, "compile_only") && py::hasattr(runtime, "enable_msprof_register_tensor")
        && !runtime.attr("compile_only").cast<bool>()
        && !runtime.attr("enable_msprof_register_tensor").cast<bool>();
    py::tuple prefix(9);
    prefix[0] = source[9]; prefix[1] = py::int_(1); prefix[2] = py::int_(1);
    prefix[3] = py::none(); prefix[4] = source[7]; prefix[5] = source[8];
    prefix[6] = py::none(); prefix[7] = py::none(); prefix[8] = py::none();
    Plan plan{source[2], source[0], source[5], source[6], prefix,
              py::int_(indices.numel()), py::int_(weight.get_device()), allowed};
    const auto limit = std::max<int64_t>(1, state->limit().cast<int64_t>());
    while (state->plans.size() >= static_cast<size_t>(limit) && !state->order.empty()) {
      state->plans.erase(state->order.front()); state->order.pop_front();
    }
    if (state->plans.emplace(*key, std::move(plan)).second) state->order.push_back(*key);
  }
  return output;
}

Tensor raw_forward(const Tensor& weight_input, const Tensor& indices_input) {
  TORCH_CHECK(state, "Ascend embedding host dispatcher is not configured");
  TORCH_CHECK(weight_input.dim() == 2, "embedding weight must be a matrix");
  TORCH_CHECK(weight_input.device().type() == c10::DeviceType::PrivateUse1
              && weight_input.device() == indices_input.device(),
              "Ascend embedding requires weight and indices on the same NPU");
  TORCH_CHECK(indices_input.scalar_type() == at::kLong || indices_input.scalar_type() == at::kInt,
              "embedding indices must be int32 or int64");
  auto weight = weight_input.contiguous();
  auto indices = indices_input.contiguous();
  const int64_t count = indices.numel(), dim = weight.size(1);
  if (!count || !dim) {
    auto shape = indices.sizes().vec(); shape.push_back(dim);
    return at::empty(shape, weight.options());
  }
  if (state->current_device().cast<int>() != weight.get_device())
    return cold_forward(weight, indices, nullptr);
  const auto wp = reinterpret_cast<uintptr_t>(weight.data_ptr());
  const auto ip = reinterpret_cast<uintptr_t>(indices.data_ptr());
  const char* debug = std::getenv("TRITON_DEBUG");
  Key key{{weight.get_device(), static_cast<int64_t>(weight.scalar_type()),
           static_cast<int64_t>(indices.scalar_type()), count, dim, weight.size(0),
           static_cast<int64_t>(wp % 16), static_cast<int64_t>(ip % 16)}, debug ? debug : "0"};
  auto found = state->plans.find(key);
  if (found == state->plans.end()) return cold_forward(weight, indices, &key);
  // A runtime call may release the GIL while another thread clears the cache.
  auto plan = found->second;
  if (!plan.raw_allowed || hooks_active(plan)) return cold_forward(weight, indices, nullptr);

  c10::SmallVector<int64_t, 8> shape(indices.sizes().begin(), indices.sizes().end());
  shape.push_back(dim);
  auto output = at::empty(shape, weight.options());
  auto stream = py::reinterpret_steal<py::object>(PyObject_CallOneArg(plan.stream.ptr(), plan.device.ptr()));
  if (!stream) throw py::error_already_set();
  auto arguments = py::reinterpret_steal<py::object>(PyTuple_New(13));
  if (!arguments) throw py::error_already_set();
  for (int index = 0; index < 9; ++index) {
    PyObject* item = index == 3 ? stream.ptr() : PyTuple_GET_ITEM(plan.prefix.ptr(), index);
    Py_INCREF(item); PyTuple_SET_ITEM(arguments.ptr(), index, item);
  }
  for (auto item : {std::make_pair(9, wp), std::make_pair(10, ip),
                    std::make_pair(11, reinterpret_cast<uintptr_t>(output.data_ptr()))}) {
    auto* value = PyLong_FromUnsignedLongLong(item.second);
    if (!value) throw py::error_already_set();
    PyTuple_SET_ITEM(arguments.ptr(), item.first, value);
  }
  Py_INCREF(plan.count.ptr()); PyTuple_SET_ITEM(arguments.ptr(), 12, plan.count.ptr());
  auto registered = py::reinterpret_steal<py::object>(PyObject_CallObject(plan.launch.ptr(), arguments.ptr()));
  if (!registered) throw py::error_already_set();
  bool active = PyLong_Check(registered.ptr()) && PyLong_AsLong(registered.ptr()) == 1;
  if (PyObject_SetAttr(state->runtime_utils.ptr(), state->profiler_key.ptr(), active ? Py_True : Py_False) < 0)
    throw py::error_already_set();
  return output;
}

class HostEmbedding : public torch::autograd::Function<HostEmbedding> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor weight, Tensor indices) {
    ctx->save_for_backward({weight, indices});
    return raw_forward(weight, indices);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
    auto saved = ctx->get_saved_variables();
    py::gil_scoped_acquire gil;
    return {state->backward(saved[0], saved[1], grads[0]).cast<Tensor>(), Tensor()};
  }
};
Tensor apply(Tensor weight, Tensor indices) { return HostEmbedding::apply(weight, indices); }
Tensor forward(Tensor weight, Tensor indices) {
  at::AutoGradMode guard(false);
  return raw_forward(weight, indices);
}
}  // namespace TORCH_EXTENSION_NAME

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("configure", &TORCH_EXTENSION_NAME::configure);
  module.def("apply", &TORCH_EXTENSION_NAME::apply);
  module.def("forward", &TORCH_EXTENSION_NAME::forward);
  module.def("clear_cache", &TORCH_EXTENSION_NAME::clear_cache);
  module.def("cache_size", &TORCH_EXTENSION_NAME::cache_size);
}
