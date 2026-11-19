cmake_minimum_required(VERSION 3.16)
project(follow_launcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

qt_add_executable(follow_launcher
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
)

target_link_libraries(follow_launcher PRIVATE Qt6::Widgets)

install(TARGETS follow_launcher RUNTIME DESTINATION bin)
