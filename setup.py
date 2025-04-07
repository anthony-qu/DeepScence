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
        ], 
    },
    install_requires=[
        "torch==2.2.2",
        "kneed",
        "protobuf>=3.9.0,<3.21.0",
        "anndata>=0.8",
        "pyyaml<=5.4.1",
    ],
    url="https://github.com/quyilong0402/DeepScence",

)
