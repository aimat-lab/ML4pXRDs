N = 4225

"""
from UNet_1DCNN import UNet


my_unet = UNet(N, 3, 1, 5, 300, output_nums=N)
model = my_unet.UNet()

model.summary()
"""

from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(N,),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation="linear",  # regression problem
)

model.summary()

