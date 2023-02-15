1. 从当前目录进入bdilab_server目录
   * cd bdilab_server
   1. 终端执行以下命令，其中model-file-path是存放漂移检测模型文件的路径
2. python -m bdilab_server --storage_uri "model-file-path" DriftDetector
3. 当bdilab-detect项目集成了新的算法,bdilab-server只需要修改Dockerfile文件里bdilab-detect的版本号即可。