from distutils.core import setup
from distutils.command.install import install as _install

class install(_install):
    '''Specialized python installer'''
    def run(self):
        _install.run(self)
        import nltk
        nltk.download('punkt')

setup(name='manatee',
    version='1.0.0',
    description='methods for anomaly notification against time-series evidence',
    packages=['manatee'],
    cmdclass={'install': install},
    install_requires=['scikit-learn',
        'numpy>=1.14.2',
        'pandas>=0.19.2',
        'matplotlib>=2.2.2',
        'tslearn>=0.1.21',
        'rrcf==0.1',
        'Simon==1.2.3',
        "Sloth==2.0.6",
        "nltk",
        "hdbscan>=0.8.18",
        'Keras >= 2.0.2, <= 2.1.6'],
    dependency_links=['git+https://github.com/NewKnowledge/tslearn@612b91cc0150d0b2c548f7426a7a4fafb864340e#egg=tslearn'
        'git+https://github.com/NewKnowledge/rrcf@1844465f28816b55ef4ef481809dcf26f968c5c3#egg=rrcf-0.1',
        'git+https://github.com/NewKnowledge/simon@e521e0d93c25b275488a98f57acf74c3144afaeb#egg=Simon-1.2.3',
        'git+https://github.com/NewKnowledge/sloth@281d54176aa3ab4a14bab43a52d141b741074d1c#egg=Sloth-2.0.6'],
    include_package_data=True)
