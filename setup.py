from setuptools import setup, find_packages
from lep import __version__

setup(
    name="LEP",
    version=__version__,             
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
        "matplotlib",
        "pandas",
        "scipy",
        "numpy"，
	    "ase"
    ],
    classifiers=[      
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
    ],
    python_requires=">=3.9",      
    entry_points={            
        'console_scripts': [
            'my_command=my_package.module:main', 
        ],
    },
)
