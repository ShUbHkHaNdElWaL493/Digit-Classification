/*
    Shubh Khandelwal
*/

#ifndef DATAPACK_H
#define DATAPACK_H

#include <fstream>
#include <vector>
#include <string>

template <typename T>
class datapack
{
    
    private:

    std::ifstream file;
    std::vector<T> data;

    public:

    void open(const std::string &);

    bool next();

    std::vector<T> get_row();

    void close();

};

#endif