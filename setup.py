from setuptools import setup
setup(name="dissim", version="0.1.1", 
      description="Discrete Simulation Optimization Based Package", 
      author="Ankush Arora", 
      author_email="nkusharoraa@gmail.com",
      packages=['dissim'],
      install_requires=["numpy", "scikit-learn", "pandas", "dask"])