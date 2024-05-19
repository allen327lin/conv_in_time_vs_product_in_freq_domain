# conv_in_time_vs_product_in_freq_domain
A simple project that compares the result of 
convolution in the time domain with 
the result of multiplication in the frequency domain. 

## Getting started
```
# Ubuntu
git clone https://github.com/allen327lin/conv_in_time_vs_product_in_freq_domain.git
cd conv_in_time_vs_product_in_freq_domain
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
### main.py
如果要儲存所有結果圖：`python3 main.py --save_img` \
如果不要儲存任何結果圖：`python3 main.py`

## License
MIT License

Copyright (c) 2024 allen327lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
