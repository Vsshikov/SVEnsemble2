{
  buildPythonApplication,
  setuptools,
  numpy,
  pandas,
  scikit-learn,
  pysam,
  igraph,
  xgboost,
  pyyaml,
}:

buildPythonApplication (_: {
  pname = "SVEnsemble2";
  version = "git";
  src = ./.;
  pyproject = true;
  build-system = [ setuptools ];
  dependencies = [
    numpy
    pandas
    scikit-learn
    pysam
    igraph
    xgboost
    pyyaml
  ];

  meta = {
    mainProgram = "svensemble";
  };
})
