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
    ├── app  # C_++ app
    │   └── desktop
    │       ├── CMakeLists.txt
    │       ├── include
    │       │   └── mainwindow.h
    │       └── src
    │           ├── main.cpp
    │           ├── maindisplay.ui
    │           └── mainwindow.cpp
    ├── common 
    │   ├── helpers # Support files for training and testing
    │   │   ├── support.py
    │   │   ├── train.py
    │   │   ├── transfer_model.py
    │   │   └── video.py
    ├── drive  
    │   ├── lanes # Lane segmentation
    │   │   ├── dataset.py
    │   │   ├── model.py
    │   ├── main.py
    │   └── paths.json
    ├── etc # Videos / Images
    ├── get_models.sh 
    ├── models 
    └── slam # Feature extraction 
        ├── extract
        │   ├── features.py
        └── vision
            └── visuals.py

# To-Do
- [x] Lane segmentation model (Quantize). 
- [ ] Road segmention model (Quantize).  
- [ ] Speed analyis via camera model (Quantize). 
- [ ] Deploy model on device. 


# License
----
MIT

