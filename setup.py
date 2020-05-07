from setuptools import setup, find_packages


setup(
    name = 'kerMIT',
    version = '2.0',
    description = 'Python implementation of kerMITencoder and kerMITviz',

    author = 'ART Uniroma2',
    
    # trying to add files...
    package_data={
        'kerMIT': ['ACTree/tree_visualizer_pyDTE/*.html', 'ACTree/tree_visualizer_pyDTE/js/*.js'] #'ACTree/**/*.html']
    },
    
    # Choose your license
    license = 'MIT',
    packages = find_packages(),
    install_requires = ['numpy', 'nltk', 'colormap', 'easydev']
)
