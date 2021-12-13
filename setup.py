from setuptools import setup, find_packages

setup(
	name='gdynet',
	version='1.0.0',
	packages=find_packages(include=['gdynet', 'tests']),
	install_requires=['numpy', 'pandas', 'matplotlib', 'tensorflow', 'pymatgen']
)
