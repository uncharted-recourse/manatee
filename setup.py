from distutils.core import setup

setup(name='manatee',
    version='1.0.0',
    description='methods for anomaly notification against time-series evidence',
    packages=['manatee'],
    install_requires=['scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'rrcf==0.1'],
    dependency_links=['git+https://github.com/NewKnowledge/rrcf@8164d6fa9787b13b575eb1eaad8d57a775adda90#egg=rrcf-0.1'],
    include_package_data=True)