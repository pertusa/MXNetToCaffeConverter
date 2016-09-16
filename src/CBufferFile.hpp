/* ----------------------------------------------------------------------------
*  MXNet auxiliary code for file handling
*  Author: Antonio Pertusa (pertusa AT ua DOT es)
*  License: GNU Public License
* ----------------------------------------------------------------------------*/

#ifndef _CBUFFERFILE_H_
#define _CBUFFERFILE_H_

#include <iostream>
#include <fstream>

using namespace std;

// Read file to buffer
class CBufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit CBufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            assert(false);
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~CBufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};

static string readAllBytes(const char *filename) // http://insanecoding.blogspot.fr/2011/11/how-to-read-in-file-in-c.html
{
   ifstream in(filename, ios::in | ios::binary);
  
   string message="Error loading binary file: ";
   message+=filename;
  
   if (!in.is_open())
   {
     cerr << message << endl; 
     exit(-1);
   }
   string contents;
   in.seekg(0, std::ios::end);
   contents.resize(in.tellg());
   in.seekg(0, std::ios::beg);
   in.read(&contents[0], contents.size());
   in.close();

   return(contents);
}

#endif
