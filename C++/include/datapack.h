/*
    Shubh Khandelwal
*/

#ifndef DATAPACK_H
#define DATAPACK_H

#include <vector>
#include <string>

template <typename T>
class datapack
{
    
    private:

    std::vector<std::vector<T>> data;

    public:

    datapack(std::vector<std::vector<T>>);

    datapack(const std::string &);

    std::vector<T> operator[](size_t);

    void print();

    size_t size();

};

#endif