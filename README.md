<img src="etc/logo_light.png" alt="logo"/>
- Using AI to help make driving safer.

## C++
```sh
   cd vigilantV2/app/desktop
   mkdir -p build && cd build
   make -j$(nproc) && ./desktop
```
## Python 
```sh
   cd vigilantV2/drive
   python main.py
```
- Visualize the model performance
```sh
  cd vigilantV2 && ./get_models.sh
  cd drive && python main.py --test True
```


# Directory Structure
------
    .
    ├── app # C++ GUI Desktop App
    │   └── desktop
    │       ├── CMakeLists.txt.user
    │       ├── include
    │       │   └── mainwindow.h
    │       └── src
    │           ├── main.cpp
    │           ├── maindisplay.ui
    │           └── mainwindow.cpp
    ├── common
    │   ├── helpers # Assist in training and testing model
    │   │   ├── support.py
    │   │   ├── train.py
    │   │   ├── transfer_model.py
    │   │   └── video.py
    ├── drive # Main Controller
    │   ├── lanes # Lane Semantic Segmentation
    │   ├── main.py
    │   └── paths.json # Paths to dataset 
    ├── etc # Images and Videos
    ├── models # Saved models


# To-Do
- [x] Lane segmentation model (Quantize). 
- [ ] Road segmention model (Quantize).  
- [ ] Speed analyis via camera model (Quantize). 
- [ ] Deploy model on device. 


# License
----
MIT

