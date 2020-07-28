rmdir /s /q _build
mkdir _build
cd _build
cmake ../ -G "Visual Studio 16" -A "x64"
cd ..