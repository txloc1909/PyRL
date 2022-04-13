import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setuptools.setup(
    name="PyRL",
    version="0.0.1",
    author="Tran Xuan Loc",
    author_email="tranxuanloc19920@gmail.com",
    decription="Python Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/txloc1909/PyRL",
    project_urls = {
        "Bug Tracker": "https://github.com/txloc1909/PyRL/issues"
    },
    license="MIT",
    packages=["pyrl"],
    install_requires=[],
)
