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
  cyvcf2,
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
    cyvcf2
  ];

  meta = {
    mainProgram = "svensemble";
  };
})
