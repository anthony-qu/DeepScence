from setuptools import setup

setup(
    name="DeepScence",
    version="1.0.0",
    description="Senescence Scoring",
    author="Anthony Qu",
    author_email="anthonyylq@gmail.com",
    packages=["DeepScence"],
    include_package_data=True,
    package_data={
        "": [
            "data/*.csv"
        ],  # Include all CSV files in data directories within any package
    },
    install_requires=[
        "torch==2.2.2",
        "kneed",
        "protobuf>=3.9.0,<3.21.0",
    ],
    url="https://github.com/quyilong0402/DeepScence",
    # license="Apache License 2.0",
    # classifiers=[
    #     "License :: OSI Approved :: Apache Software License",
    #     "Topic :: Scientific/Engineering :: Artificial Intelligence",
    #     "Programming Language :: Python :: 3.5",
    # ],
)
