from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='CQS',
    url='https://github.com/kemperlab/cartan-quantum-synthesizer',
    author='Thomas Steckmann, Efekan Kokcu',
    author_email='tmsteckm@ncsu.edu, ekokcu@ncsu.edu',
    # Needed to actually package something
    packages=['src','util'],
    project_urls={
        "Documtation": "https://kemperlab.github.io/cartan-quantum-synthesizer/",
    },
    # Needed for dependencies
    install_requires=['numpy','scipy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Implementation of the Cartan Decomposition for generating time evolution circuits on lattice spin models',
)