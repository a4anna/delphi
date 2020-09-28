python setup.py install 
systemctl restart delphi
systemctl is-active delphi >/dev/null 2>&1 && echo "delphi is active" || echo "delphi NOT active"
