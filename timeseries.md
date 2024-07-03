https://tensorflow.google.cn/tutorials/structured_data/time_series#feature_engineering

### pandas pop和push操作

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

这里timestamp_s是Series属性的变量，可以直接使用numpy计算push进df里

### 训练集验证集测试集划分不随机，two reason：

It ensures that chopping the data into windows of consecutive samples is still possible.

It ensures that the validation/test results are **more realistic**, being evaluated on the data collected after the model was trained. ？

### 归一化中平均值与标准差只能用训练集计算 ？

### _ 占位符，表示忽略返回值
如：

    1.csv_path, _ = os.path.splitext(zip_path)  
    本来会返回两个值只取第一个
    2._ = plt.xlabel('Frequency (log scale)')
    本来这个函数会返回一个文本值？



### *星号操作符 
可用于解压参数列表或将任意个参数导入函数中

https://www.runoob.com/w3cnote/python-one-and-two-star.html