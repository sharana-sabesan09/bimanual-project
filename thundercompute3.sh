sudo apt-get install -y xvfb
cat > run.sh << 'EOF'
#!/usr/bin/env bash
export PYOPENGL_PLATFORM=osmesa
exec xvfb-run -a -s "-screen 0 1280x1024x24" python "$@"
EOF
chmod +x run.sh