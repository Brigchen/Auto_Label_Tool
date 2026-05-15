# Auto_Label_Tool（多功能自动标注工具）

### Jixin CHEN, from Xiamen University
email: brigchen@gmail.com

### [<u>中文</u>](./readme_CN.md)

### 简介：

在经典的labelImg(https://github.com/tzutalin/labelImg)的基础上，增加了多种标注工具, 形成自动标注工具包。新增功能包含如下：

- **`TOOL LIST`**：
- [x] **自动标注**：基于yolov8的模型自动标注，单类群与多类群两个模式
- [x]  **更新模型**：基于新标注的数据，训练并更新自动标注模型
- [x] **视频追踪标注**：利用opencv的追踪功能，自动标注视频数据
- [x] **放大镜**：局部放大，对小目标的标注有帮助，可以关闭
- [x] 其他辅助工具：类别筛选/重命名/统计、标注文件属性校正、视频提取/合成、图片重命名等，可以利用查询系统查看详细信息，欢迎体验


## 安装步骤：

1. 复制：

   解压缩包至工作目录下
   ```

2. 安装依赖工具包：

   ```
   cd 工作目录
   pip install -r requirements.txt
   ```

   
3. 准备yolov8模型并放置在如下位置，

   ```
   ultralytics/weights/{your_model_weight.pt} 
   ```

5. 打开软件，开始标注或模型训练 

   ```
   python ALT.py
   ```


## 设置快捷方式[非必须]

**Windows用户:**

桌面创建Auto_Label_Tool.bat（可以新建文本文件，然后把后缀.txt改成.bat）,右键用文本编辑器打开，键入下面内容(不一定是D盘，根据实际输入)：

```
cd {path to your ALT folder}
start python ALT.py
exit
```

双击Auto_Label_Tool.bat即可打开标注软件。



