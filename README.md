# Triton-Ascend Agent Development Kit

## What is this?

This package is for your AI agent, which will try to help you develop kernels in the Triton-Ascend dialect for subsequent execution on the Ascend NPU.
Keep in mind that Triton-Ascend, as well as the NPU itself, are constantly being improved and changed, and something proposed by your AI agent when using this package may not be suitable for your specific environment.

Keep the following resources in focus:

[https://www.hiascend.com/en/document](https://www.hiascend.com/en/document)

[https://support.huawei.com/enterprise/en/category/ascend-computing-pid-1557196528909?submodel=doc](https://support.huawei.com/enterprise/en/category/ascend-computing-pid-1557196528909?submodel=doc)

[https://github.com/Ascend](https://github.com/Ascend)

[https://triton-ascend.readthedocs.io/en/latest/index.html](https://triton-ascend.readthedocs.io/en/latest/index.html)

[https://github.com/triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend)

[https://gitcode.com/Ascend/triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels)

[https://gitcode.com/AndyCandy/triton-ascend-ops](https://gitcode.com/AndyCandy/triton-ascend-ops)


## Contents

- `AGENTS.md` - project instructions for Triton-Ascend work.
- `triton-ascend-dev-doc.md` - the guide index and agent loading rules.
- `triton-ascend-dev-doc/` - task-oriented development references.
-  `skills/` -24 task-specific Agent Skills.

## How to install

- Get the corresponding archive from the Releases of this repository.
- Or, run the `build-workspace-agent-specific-kit.sh` script, and then take the required archive from the `dist/` folder
- Just unpack the corresponding archive into the agent’s working folder.
  
## License

Licensed under the [Apache License 2.0](LICENSE).
