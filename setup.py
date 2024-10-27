from setuptools import setup, find_packages

setup(
    name='hybrid_movie_recommender',
    version='0.1',
    author='BKP',
    description='A hybrid movie recommender system using machine learning models.',
    packages=find_packages(),
    install_requires=[
        'pandas==1.3.5',
        'scikit-learn==1.0.2',
        'joblib==1.1.0',
        'scipy==1.7.3',
        'flask==2.0.2'
    ],
    entry_points={
        'console_scripts': [
            'run-app=app:main',
        ],
    },
)
