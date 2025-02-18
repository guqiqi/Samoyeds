pybind11_add_module(samoyeds_kernel MODULE
        api/samoyeds_mod.cu
)

target_include_directories(samoyeds_kernel PUBLIC
        ${SAMOYEDS_SRC_DIR}
)

target_link_libraries(samoyeds_kernel
        PRIVATE ${TORCH_LIBRARIES}
        PRIVATE ${TORCH_PYTHON_LIBRARY}
)