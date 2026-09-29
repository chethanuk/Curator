# Simulate a nemo_curator install whose wheel lacks xenna_grafana_dashboard.json
# (issue #2189) and provision Grafana into a fresh metrics dir.
import os, tempfile
from nemo_curator.metrics import utils
pkg = os.path.join(os.path.dirname(utils.__file__), "xenna_grafana_dashboard.json")
print("packaged dashboard present:", os.path.isfile(pkg))
d = tempfile.mkdtemp()
ini = utils.write_grafana_configs(3000, 9090, metrics_dir=d)
dash = os.path.join(d, "grafana", "dashboards")
print("grafana.ini written:", os.path.isfile(ini))
print("xenna dashboard provisioned:", os.path.isfile(os.path.join(dash, "xenna_grafana_dashboard.json")))
