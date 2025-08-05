from setuptools import setup, find_packages

setup(
    name="LEP",
    version="0.1.0",             
    packages=find_packages(),   
    description="LEP",  
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown", 
    url="https://https://github.com/XZ-HaN/LEP", 
    author="HaN",  
    author_email="zybaozi@sjtu.edu.cn",   
    license="GPL-3.0",   
    install_requires=[         
        "requests>=2.25.1",    
        "numpy",
        "matplotlib==3.7.1"
    ],
    classifiers=[      
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GLP-3 License",
    ],
    python_requires=">=3.9",      
    entry_points={            
        'console_scripts': [
            'my_command=my_package.module:main', 
        ],
    },
)