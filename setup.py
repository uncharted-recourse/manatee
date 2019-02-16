from distutils.core import setup

setup(name='manatee',
    version='1.0.0',
    description='methods for anomaly notification against time-series evidence',
    packages=['manatee'],
    install_requires=['scikit-learn',
        'numpy>=1.14.2',
        'pandas>=0.19.2',
        'matplotlib>=2.2.2',
        'tslearn>=0.1.21',
        'rrcf==0.1'],
    dependency_links=['git+https://github.com/NewKnowledge/rrcf@1844465f28816b55ef4ef481809dcf26f968c5c3#egg=rrcf-0.1'],
    include_package_data=True)
