from setuptools import setup

setup(
    name='CQS',
    url='https://github.com/kemperlab/cartan-quantum-synthesizer',
    author='Thomas Steckmann, Efekan Kokcu',
    author_email='tmsteckm@ncsu.edu, ekokcu@ncsu.edu',
    packages=['CQS.util', 'CQS'],
    project_urls={
        "Documentation": "https://kemperlab.github.io/cartan-quantum-synthesizer/",
    },
    install_requires=['numpy','scipy'],
    version='0.2',
    license='BSD-2-Clause Plus Patent License',
    description='Implementation of the Cartan Decomposition for generating time evolution circuits on lattice spin models',
    entry_points={
            'qiskit.synthesis': [
                'PauliEvolution.cartan = CQS.plugin.cartan_plugin:CartanPlugin',
            ],
    },
)
