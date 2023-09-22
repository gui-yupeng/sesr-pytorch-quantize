# pytorch神经网络量化框架
## 描述：
    test.py使用models/sesr.py,完成初步的量化测试和激活的校准
    sim.py使用models/sesr_sim.py,完全模拟硬件，测psnr
    define.py 配置量化框架

## 使用：
```
python test.py
python sim.py
//导出测试激励
python output.py
```
