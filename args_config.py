args.count = 1000

# 字长度
# -w 10
args.length = range(1, 10)

# 随机字长度
# -r
args.random = True

# 图像高度 已改为默认48
# -f 48
args.format  check

# 旋转角度
# -k 10
args.skew_angle = range(0, 10)
# -rk
args.random_skew = True

# 模糊度水平(最大值范围待测)
# -bl 3
args.blur = 3
# 模糊度水平随机 ( 0 --> args.blur)
# -rbl
args.random_blur = True


# 背景种类 （需要补充背景图片数量）
# 先平均生成，后多生成部分某类，或增加某些类
# -b 0,1,2,3 (需要loop!)
args.background = range(0, 4)

# 对齐
# 平均生成
# -al 0,1,2 (需要loop!)
args.alignment = range(0, 3)


# 边界
# 定义多边界
# -m (5,5,5,5) 需要多个loop! 上下左右四个方向的margin
args.aligment = (range(0, 5), range(0, 5), range(0, 5), range(0, 5))
