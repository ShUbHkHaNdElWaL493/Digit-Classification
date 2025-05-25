/*
    Shubh Khandelwal
*/

#include "datapack.h"
#include <iostream>
#include <fstream>
#include <sstream>

template class datapack<char>;
template class datapack<double>;
template class datapack<float>;
template class datapack<int>;
template class datapack<std::string>;

template <typename T>
void datapack<T>::open(const std::string &path)
{

    this->file.open(path);
    if (!this->file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

}

template <typename T>
bool datapack<T>::next()
{

    std::cerr << "Error: Could not read line." << std::endl;
    exit(EXIT_FAILURE);

}

template <>
bool datapack<char>::next()
{

    this->data.clear();
    
    std::string line;
    if (!std::getline(this->file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        this->data.push_back(cell[0]);
    }

    return true;

}

template <>
bool datapack<double>::next()
{

    this->data.clear();
    
    std::string line;
    if (!std::getline(this->file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        this->data.push_back(std::stod(cell));
    }

    return true;

}

template <>
bool datapack<float>::next()
{

    this->data.clear();
    
    std::string line;
    if (!std::getline(this->file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        this->data.push_back(std::stof(cell));
    }

    return true;

}

template <>
bool datapack<int>::next()
{

    this->data.clear();

    std::string line;
    if (!std::getline(this->file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        this->data.push_back(std::stoi(cell));
    }

    return true;

}

template <>
bool datapack<std::string>::next()
{

    this->data.clear();

    std::string line;
    if (!std::getline(this->file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        this->data.push_back(cell);
    }

    return true;

}

template <typename T>
std::vector<T> datapack<T>::get_row()
{
    return this->data;
}

template <typename T>
void datapack<T>::close()
{

    this->file.close();

}