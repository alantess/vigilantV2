#! /bin/bash

"/usr/bin/cmake" --build /home/alan/Code/vigilantV2/app/build-Vigilant --target all Vigilant_prepare_apk_dir

"/home/alan/Qt/6.1.0/gcc_64/bin/androiddeployqt" --input /home/alan/Code/vigilantV2/app/build-Vigilant/android-Vigilant-deployment-settings.json --output /home/alan/Code/vigilantV2/app/build-Vigilant/android-build --android-platform android-30 --jdk /usr/lib/jvm/java-11-openjdk-amd64/ --gradle

/home/alan/Qt/6.1.0/gcc_64/bin/androiddeployqt --verbose --output /home/alan/Code/vigilantV2/app/build-Vigilant/android-build --no-build --input /home/alan/Code/vigilantV2/app/build-Vigilant/android-Vigilant-deployment-settings.json --gradle --reinstall --device RF8N40GZRQB
