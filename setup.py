from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="Cryo-TTT",
        version=1.0,
        description="Test-time training for Cryo-ET segmentation",
        author="Diyor Khayrutdinov",
        author_email="diyor.khayrutdinov@tum.de",
        license="Apache License 2.0",
        packages=find_packages(exclude=["tools", "data", "output"]),
    )
