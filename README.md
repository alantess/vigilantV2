<img src="etc/logo_light.png" alt="logo"/>
- Using AI to help make driving safer.

# Execute
## Download Models
```sh
   cd vigilantV2/models 
   ./get_models.sh 
  ```
## C++
```sh
   cd vigilantV2/app/desktop
   mkdir -p build && cd build
   make -j$(nproc) && ./desktop
  ```
## Python 
```sh
   cd vigilantV2/common
   python main.py
  ```

# Directory Structure
------
    .
     ├── common          #  Main control for all models 
     ├── app             #  Desktop / Android Application 
     ├── Etc             #  Random Files, Images, Gifs

# To-Do
- [x] Lane segmentation model (Quantize). 
- [ ] Road segmention model (Quantize).  
- [ ] Speed analyis via camera model (Quantize). 
- [ ] Deploy model on device. 


# License
----
MIT

