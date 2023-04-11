1. 从当前目录进入bdilab_server目录
   * cd bdilab_server
2. 终端执行以下命令，其中model-file-path是存放漂移检测模型文件的路径
3. python -m bdilab_server --storage_uri "model-file-path" DriftDetector 
4. 当bdilab-detect项目集成了新的算法,bdilab-server只需要修改Dockerfile文件里bdilab-detect的版本号即可。
5. 启动示例
   python -m bdilab_server --storage_uri "/Users/treasures/Downloads/model/sddm" DriftDetector --p_val 0.07 --drift_batch_size 3