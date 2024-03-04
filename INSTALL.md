# Installing Conda Environment for Real-World Applications
Installation guidance for sim-pretraining and real-world finetuning on quadruped robots.
# 0 create python/pytorch env (Python 3.6, 3.7, or 3.8 (3.8 recommended))
```
conda create -n unio4-real python=3.8
conda activate unio4-real
```
# 1 Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (I didn't test the history version) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`

# 2 Install legged_gym
   - `cd ../legged_gym && pip install -e .`

# 3 sdk building
   - `cd ../go1_sdk`
   - `mkdir build && cd ./build`
   - `cmake ..`
   - `make`
# 4 hardward T265 Installation
Create a new script：

`nano realsense.sh`

Copy the following to it:

```
export REALSENSE_SOURCE_DIR=$HOME/projects/librealsense/
sudo apt-get install guvcview git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
git clone https://github.com/IntelRealSense/librealsense.git $REALSENSE_SOURCE_DIR
mkdir $REALSENSE_SOURCE_DIR/build
cd $REALSENSE_SOURCE_DIR/build
export REALSENSE_INSTALL_PREFIX=/opt/realsense
sudo mkdir -p $REALSENSE_INSTALL_PREFIX; 
sudo chown $USER:$USER -R $REALSENSE_INSTALL_PREFIX # not relay needed -> you could run _make_ followed by _sudo make install_
cmake ../ -DFORCE_RSUSB_BACKEND=true -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DCMAKE_INSTALL_PREFIX=$REALSENSE_INSTALL_PREFIX
make install
sudo sh -c "echo $REALSENSE_INSTALL_PREFIX/lib > /etc/ld.so.conf.d/realsense.conf"
sudo ldconfig
cd ~/projects/librealsense/
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules && sudo udevadm control --reload-rules && udevadm trigger
echo "export realsense2_DIR=/opt/realsense/lib/cmake/realsense2" >> ~/.bashrc
reboot
```
Run:
```
chmod +x realsense.sh
./realsense.sh
```