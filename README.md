# yolo_trashdetect
Darknet YOLOv4-tiny垃圾检测分类，2021年工训赛项目。项目源代码注释均保留，易于阅读。</br>
本项目代码主要使用Darknet训练得到的YOLOv4-tiny模型对垃圾进行识别，也提供Resnet模型识别和京东在线识别模式，默认适配Jetson Nano平台（支持GPU加速），可通过修改驱动层代码改为支持树莓派等其他硬件。</br>
项目使用Tkinter作为GUI框架，具备垃圾图片显示、识别显示、满载检测等功能</br>
项目代码在实际使用时需要配合使用串口连接的下位机使用，下位机可以用Arduino/STM32等平台进行制作，主要进行满载检测和舵机驱动。本项目开源不含此部分，但可通过阅读代码自行实现下位机（协议很简单）</br>
项目在GPLv3 Licence下开源，不提供Warranty！</br>
特别声明：项目中上传的mp4视频文件版权归对应创作者所有，在本项目中仅用作示例！如果您是视频文件的原作者且认为收到侵权，请提出issue，我会尽快删除/替换示例视频文件。
# Hardware design
本项目电路部分的硬件设计可以从以下链接中找到：
https://oshwhub.com/GK1999/usb-pd-dian-yuan-mu-kuai
