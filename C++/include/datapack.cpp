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
datapack<T>::datapack(std::vector<std::vector<T>> data)
{
    this->data = data;
}

template <typename T>
datapack<T>::datapack(const std::string &path)
{
    std::cerr << "Error: Unsupported type for datapack.\n";
    exit(EXIT_FAILURE);
}

template <>
datapack<char>::datapack(const std::string &path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::vector<char> row;
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell[0]);
        }
        this->data.push_back(row);

    }

    file.close();

}

template <>
datapack<double>::datapack(const std::string &path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stod(cell));
        }
        this->data.push_back(row);

    }

    file.close();

}

template <>
datapack<float>::datapack(const std::string &path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::vector<float> row;
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stof(cell));
        }
        this->data.push_back(row);

    }

    file.close();

}

template <>
datapack<int>::datapack(const std::string &path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::vector<int> row;
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stoi(cell));
        }
        this->data.push_back(row);

    }

    file.close();

}

template <>
datapack<std::string>::datapack(const std::string &path)
{

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(file, line))
    {

        std::stringstream ss(line);
        std::vector<std::string> row;
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }
        this->data.push_back(row);

    }

    file.close();

}

template <typename T>
std::vector<T> datapack<T>::operator[](size_t index)
{

    if (index >= this->data.size())
    {
        throw std::out_of_range("Index out of range");
    }

    return this->data[index];

}

template <typename T>
void datapack<T>::print()
{
    for (const std::vector<T>& row : data)
    {
        for (const T& cell : row)
        {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
size_t datapack<T>::size()
{
    return this->data.size();
}