#include "MainWindow.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSplitter>
#include <QTextCursor>
#include <QVBoxLayout>

namespace {

QDoubleSpinBox *makeSpin(double min, double max, double step, double value, int decimals = 3)
{
    auto *s = new QDoubleSpinBox;
    s->setRange(min, max);
    s->setSingleStep(step);
    s->setDecimals(decimals);
    s->setValue(value);
    return s;
}

QString quoteArg(const QString &arg)
{
    if (arg.contains(QLatin1Char(' ')) || arg.contains(QLatin1Char('"'))) {
        QString escaped = arg;
        escaped.replace(QLatin1Char('"'), QStringLiteral("\\\""));
        return QStringLiteral("\"%1\"").arg(escaped);
    }
    return arg;
}

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_process(new QProcess(this))
    , m_killTimer(new QTimer(this))
{
    setWindowTitle(tr("Cobotique — follow_object"));
    resize(920, 780);

    m_killTimer->setSingleShot(true);
    connect(m_killTimer, &QTimer::timeout, this, &MainWindow::onKillTimeout);

    m_process->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_process, &QProcess::readyRead, this, &MainWindow::onProcessReadyRead);
    connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::onProcessFinished);
    connect(m_process, &QProcess::errorOccurred, this, &MainWindow::onProcessError);

    buildUi();
    onScenarioChanged();
    refreshCommandPreview();
    setRunning(false);
}

MainWindow::~MainWindow()
{
    stopProcess(true);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_process->state() != QProcess::NotRunning) {
        stopProcess(true);
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::buildUi()
{
    auto *central = new QWidget(this);
    setCentralWidget(central);
    auto *root = new QVBoxLayout(central);

    auto *splitter = new QSplitter(Qt::Vertical);
    root->addWidget(splitter, 1);

    auto *topScroll = new QScrollArea;
    topScroll->setWidgetResizable(true);
    auto *top = new QWidget;
    auto *topLayout = new QVBoxLayout(top);
    topLayout->addWidget(buildScenarioBox());
    topLayout->addWidget(buildTargetBox());
    topLayout->addWidget(buildUsefulBox());
    topLayout->addWidget(buildLabBox());
    topLayout->addWidget(buildUrsimBox());
    topLayout->addWidget(buildAdvancedBox());
    topLayout->addStretch(1);
    topScroll->setWidget(top);
    splitter->addWidget(topScroll);

    auto *bottom = buildRunBox();
    splitter->addWidget(bottom);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
}

QWidget *MainWindow::buildScenarioBox()
{
    auto *box = new QGroupBox(tr("Scénario"));
    auto *layout = new QHBoxLayout(box);
    m_scDetection = new QRadioButton(tr("Détection seule"));
    m_scLab = new QRadioButton(tr("Robot lab"));
    m_scUrsim = new QRadioButton(tr("URSim"));
    m_scDetection->setChecked(true);
    m_scenarioGroup = new QButtonGroup(this);
    m_scenarioGroup->addButton(m_scDetection, 0);
    m_scenarioGroup->addButton(m_scLab, 1);
    m_scenarioGroup->addButton(m_scUrsim, 2);
    layout->addWidget(m_scDetection);
    layout->addWidget(m_scLab);
    layout->addWidget(m_scUrsim);
    layout->addStretch(1);
    connect(m_scenarioGroup, &QButtonGroup::idClicked, this, [this](int) {
        onScenarioChanged();
        refreshCommandPreview();
    });
    return box;
}

QWidget *MainWindow::buildTargetBox()
{
    auto *box = new QGroupBox(tr("Cible"));
    auto *layout = new QFormLayout(box);

    auto *modeRow = new QHBoxLayout;
    m_tgColor = new QRadioButton(tr("Couleur seule"));
    m_tgShape = new QRadioButton(tr("Couleur + forme (--shape)"));
    m_tgColor->setChecked(true);
    m_targetGroup = new QButtonGroup(this);
    m_targetGroup->addButton(m_tgColor, 0);
    m_targetGroup->addButton(m_tgShape, 1);
    modeRow->addWidget(m_tgColor);
    modeRow->addWidget(m_tgShape);
    modeRow->addStretch(1);
    layout->addRow(tr("Mode"), modeRow);

    m_color = new QComboBox;
    m_color->addItems({QStringLiteral("red"), QStringLiteral("yellow"),
                       QStringLiteral("green"), QStringLiteral("blue")});
    m_color->setCurrentText(QStringLiteral("blue"));
    layout->addRow(tr("Couleur (--color)"), m_color);

    auto *shapeRow = new QHBoxLayout;
    m_shapePath = new QLineEdit;
    const QString shapeHint = tr("vide = photo la plus récente dans data/");
    m_shapePath->setPlaceholderText(shapeHint);
    m_shapePath->setMinimumWidth(m_shapePath->fontMetrics().horizontalAdvance(shapeHint) + 24);
    m_browseShape = new QPushButton(tr("…"));
    m_browseShape->setFixedWidth(36);
    shapeRow->addWidget(m_shapePath, 1);
    shapeRow->addWidget(m_browseShape);
    layout->addRow(tr("Photo (--shape)"), shapeRow);

    connect(m_targetGroup, &QButtonGroup::idClicked, this, [this](int) {
        onTargetChanged();
        refreshCommandPreview();
    });
    connect(m_color, &QComboBox::currentTextChanged, this, [this](const QString &) {
        refreshCommandPreview();
    });
    connect(m_shapePath, &QLineEdit::textChanged, this, [this](const QString &) {
        refreshCommandPreview();
    });
    connect(m_browseShape, &QPushButton::clicked, this, &MainWindow::onBrowseShape);
    onTargetChanged();
    return box;
}

QWidget *MainWindow::buildUsefulBox()
{
    m_usefulBox = new QGroupBox(tr("Réglages"));
    auto *layout = new QFormLayout(m_usefulBox);

    m_camera = new QLineEdit(QStringLiteral("0"));
    m_camera->setPlaceholderText(tr("0, chemin, ou realsense"));
    layout->addRow(tr("Caméra (--camera)"), m_camera);

    m_gain = makeSpin(0.05, 10.0, 0.1, 1.0, 2);
    layout->addRow(tr("Gain (--gain)"), m_gain);

    m_feedforward = makeSpin(0.0, 1.0, 0.05, 0.3, 2);
    layout->addRow(tr("Feedforward"), m_feedforward);

    m_maxSpeed = makeSpin(0.005, 0.08, 0.005, 0.02, 3);
    layout->addRow(tr("Vitesse max (m/s)"), m_maxSpeed);

    m_radius = makeSpin(0.05, 1.0, 0.05, 0.25, 2);
    layout->addRow(tr("Rayon (--radius)"), m_radius);

    m_followDepth = new QCheckBox(tr("Suivre aussi en profondeur (--follow-depth)"));
    layout->addRow(QString(), m_followDepth);

    m_targetDistance = makeSpin(0.2, 2.0, 0.05, 0.5, 2);
    layout->addRow(tr("Distance cible (m)"), m_targetDistance);

    m_depthGain = makeSpin(0.05, 10.0, 0.1, 1.0, 2);
    layout->addRow(tr("Gain profondeur"), m_depthGain);

    connect(m_camera, &QLineEdit::textChanged, this, [this](const QString &) {
        refreshCommandPreview();
    });
    for (auto *s : {m_gain, m_feedforward, m_maxSpeed, m_radius, m_targetDistance, m_depthGain}) {
        connect(s, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double) {
            refreshCommandPreview();
        });
    }
    connect(m_followDepth, &QCheckBox::toggled, this, &MainWindow::onFollowDepthToggled);
    onFollowDepthToggled(false);
    return m_usefulBox;
}

QWidget *MainWindow::buildLabBox()
{
    m_labBox = new QGroupBox(tr("Robot lab"));
    auto *layout = new QFormLayout(m_labBox);

    m_ip = new QLineEdit(QStringLiteral("100.65.63.93"));
    layout->addRow(tr("IP (--ip)"), m_ip);

    auto *poseRow = new QHBoxLayout;
    m_poseHome = new QRadioButton(tr("--home"));
    m_poseStartHere = new QRadioButton(tr("--start-here"));
    m_poseHome->setChecked(true);
    m_poseGroup = new QButtonGroup(this);
    m_poseGroup->addButton(m_poseHome, 0);
    m_poseGroup->addButton(m_poseStartHere, 1);
    poseRow->addWidget(m_poseHome);
    poseRow->addWidget(m_poseStartHere);
    poseRow->addStretch(1);
    layout->addRow(tr("Pose"), poseRow);

    m_calibMode = new QComboBox;
    m_calibMode->addItem(tr("Charger calib (--load-calib)"), QStringLiteral("load"));
    m_calibMode->addItem(tr("Calibrer objet (--calibrate)"), QStringLiteral("calibrate"));
    m_calibMode->addItem(tr("Calibrer scène (--calibrate-scene)"), QStringLiteral("scene"));
    m_calibMode->addItem(tr("Aucune"), QStringLiteral("none"));
    layout->addRow(tr("Calibration"), m_calibMode);

    connect(m_ip, &QLineEdit::textChanged, this, [this](const QString &) {
        refreshCommandPreview();
    });
    connect(m_poseGroup, &QButtonGroup::idClicked, this, [this](int) {
        refreshCommandPreview();
    });
    connect(m_calibMode, &QComboBox::currentIndexChanged, this, [this](int) {
        refreshCommandPreview();
    });
    return m_labBox;
}

QWidget *MainWindow::buildUrsimBox()
{
    m_ursimBox = new QGroupBox(tr("URSim"));
    auto *layout = new QHBoxLayout(m_ursimBox);
    m_ursimSim = new QRadioButton(tr("--sim"));
    m_ursimWebcam = new QRadioButton(tr("--webcam-table"));
    m_ursimSim->setChecked(true);
    m_ursimGroup = new QButtonGroup(this);
    m_ursimGroup->addButton(m_ursimSim, 0);
    m_ursimGroup->addButton(m_ursimWebcam, 1);
    layout->addWidget(m_ursimSim);
    layout->addWidget(m_ursimWebcam);
    layout->addStretch(1);
    connect(m_ursimGroup, &QButtonGroup::idClicked, this, [this](int) {
        refreshCommandPreview();
    });
    return m_ursimBox;
}

QWidget *MainWindow::buildAdvancedBox()
{
    m_advancedBox = new QGroupBox(tr("Avancé"));
    m_advancedBox->setCheckable(true);
    m_advancedBox->setChecked(false);
    auto *layout = new QFormLayout(m_advancedBox);

    m_height = makeSpin(0.05, 1.0, 0.05, 0.30, 2);
    layout->addRow(tr("Hauteur (--height)"), m_height);

    m_minZ = makeSpin(0.0, 0.5, 0.01, 0.10, 2);
    layout->addRow(tr("Z min (--min-z)"), m_minZ);

    m_calibStep = makeSpin(0.01, 0.15, 0.01, 0.05, 2);
    layout->addRow(tr("Pas calib (--calib-step)"), m_calibStep);

    m_objectDistance = makeSpin(0.1, 2.0, 0.05, 0.5, 2);
    layout->addRow(tr("Distance objet (--object-distance)"), m_objectDistance);

    auto *homeRow = new QHBoxLayout;
    m_homeFile = new QLineEdit(QStringLiteral("data/follow_home.json"));
    auto *browseHome = new QPushButton(tr("…"));
    browseHome->setFixedWidth(36);
    homeRow->addWidget(m_homeFile);
    homeRow->addWidget(browseHome);
    layout->addRow(tr("Fichier home"), homeRow);
    connect(browseHome, &QPushButton::clicked, this, &MainWindow::onBrowseHomeFile);

    auto *calibRow = new QHBoxLayout;
    m_calibFile = new QLineEdit(QStringLiteral("data/follow_calibration.json"));
    auto *browseCalib = new QPushButton(tr("…"));
    browseCalib->setFixedWidth(36);
    calibRow->addWidget(m_calibFile);
    calibRow->addWidget(browseCalib);
    layout->addRow(tr("Fichier calib"), calibRow);
    connect(browseCalib, &QPushButton::clicked, this, &MainWindow::onBrowseCalibFile);

    m_duration = makeSpin(0.0, 3600.0, 1.0, 0.0, 0);
    layout->addRow(tr("Durée s (--duration, 0 = ∞)"), m_duration);

    m_headless = new QCheckBox(tr("--headless"));
    layout->addRow(QString(), m_headless);

    m_simMountEnable = new QCheckBox(tr("Utiliser --sim-mount"));
    layout->addRow(QString(), m_simMountEnable);
    m_simMountYaw = makeSpin(-180.0, 180.0, 5.0, 0.0, 1);
    layout->addRow(tr("sim-mount yaw (deg)"), m_simMountYaw);
    m_simMountFlip = new QComboBox;
    m_simMountFlip->addItem(QStringLiteral("0"), 0);
    m_simMountFlip->addItem(QStringLiteral("1"), 1);
    layout->addRow(tr("sim-mount flip"), m_simMountFlip);

    auto *repoRow = new QHBoxLayout;
    m_repoRoot = new QLineEdit;
    m_repoRoot->setPlaceholderText(tr("auto = parent de gui/"));
    auto *browseRepo = new QPushButton(tr("…"));
    browseRepo->setFixedWidth(36);
    repoRow->addWidget(m_repoRoot);
    repoRow->addWidget(browseRepo);
    layout->addRow(tr("Racine du dépôt"), repoRow);
    connect(browseRepo, &QPushButton::clicked, this, &MainWindow::onBrowseRepoRoot);

    auto *uvRow = new QHBoxLayout;
    m_uvPath = new QLineEdit(QStringLiteral("uv"));
    auto *browseUv = new QPushButton(tr("…"));
    browseUv->setFixedWidth(36);
    uvRow->addWidget(m_uvPath);
    uvRow->addWidget(browseUv);
    layout->addRow(tr("Binaire uv"), uvRow);
    connect(browseUv, &QPushButton::clicked, this, &MainWindow::onBrowseUv);

    auto refresh = [this](auto) { refreshCommandPreview(); };
    for (auto *s : {m_height, m_minZ, m_calibStep, m_objectDistance, m_duration, m_simMountYaw}) {
        connect(s, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, refresh);
    }
    connect(m_homeFile, &QLineEdit::textChanged, this, refresh);
    connect(m_calibFile, &QLineEdit::textChanged, this, refresh);
    connect(m_repoRoot, &QLineEdit::textChanged, this, refresh);
    connect(m_uvPath, &QLineEdit::textChanged, this, refresh);
    connect(m_headless, &QCheckBox::toggled, this, [this](bool) { refreshCommandPreview(); });
    connect(m_simMountEnable, &QCheckBox::toggled, this, [this](bool) { refreshCommandPreview(); });
    connect(m_simMountFlip, &QComboBox::currentIndexChanged, this, [this](int) {
        refreshCommandPreview();
    });
    connect(m_advancedBox, &QGroupBox::toggled, this, [this](bool) { refreshCommandPreview(); });

    return m_advancedBox;
}

QWidget *MainWindow::buildRunBox()
{
    auto *box = new QWidget;
    auto *layout = new QVBoxLayout(box);
    layout->setContentsMargins(0, 0, 0, 0);

    layout->addWidget(new QLabel(tr("Commande")));
    m_commandPreview = new QLineEdit;
    m_commandPreview->setReadOnly(true);
    layout->addWidget(m_commandPreview);

    auto *buttons = new QHBoxLayout;
    m_launch = new QPushButton(tr("Lancer"));
    m_stop = new QPushButton(tr("Arrêter"));
    m_clearLogs = new QPushButton(tr("Effacer les logs"));
    buttons->addWidget(m_launch);
    buttons->addWidget(m_stop);
    buttons->addWidget(m_clearLogs);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    m_logs = new QPlainTextEdit;
    m_logs->setReadOnly(true);
    m_logs->setPlaceholderText(tr("Sortie de follow_object…"));
    m_logs->setMinimumHeight(160);
    layout->addWidget(m_logs, 1);

    connect(m_launch, &QPushButton::clicked, this, &MainWindow::onLaunch);
    connect(m_stop, &QPushButton::clicked, this, &MainWindow::onStop);
    connect(m_clearLogs, &QPushButton::clicked, this, &MainWindow::onClearLogs);
    return box;
}

MainWindow::Scenario MainWindow::currentScenario() const
{
    if (m_scLab->isChecked())
        return Scenario::LabRobot;
    if (m_scUrsim->isChecked())
        return Scenario::URSim;
    return Scenario::Detection;
}

MainWindow::Target MainWindow::currentTarget() const
{
    return m_tgShape->isChecked() ? Target::Shape : Target::ColorOnly;
}

MainWindow::PoseMode MainWindow::currentPoseMode() const
{
    return m_poseStartHere->isChecked() ? PoseMode::StartHere : PoseMode::Home;
}

MainWindow::CalibMode MainWindow::currentCalibMode() const
{
    const QString id = m_calibMode->currentData().toString();
    if (id == QLatin1String("calibrate"))
        return CalibMode::Calibrate;
    if (id == QLatin1String("scene"))
        return CalibMode::CalibrateScene;
    if (id == QLatin1String("none"))
        return CalibMode::None;
    return CalibMode::Load;
}

MainWindow::UrsimMode MainWindow::currentUrsimMode() const
{
    return m_ursimWebcam->isChecked() ? UrsimMode::WebcamTable : UrsimMode::Sim;
}

QString MainWindow::repoRoot() const
{
    const QString custom = m_repoRoot->text().trimmed();
    if (!custom.isEmpty())
        return QDir(custom).absolutePath();

    // Executable lives in gui/build/…; walk up until we find project/follow_object.py
    QDir dir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 8; ++i) {
        if (QFileInfo(dir.filePath(QStringLiteral("project/follow_object.py"))).isFile())
            return dir.absolutePath();
        if (!dir.cdUp())
            break;
    }
    // Fallback: gui/ is next to project/
    dir = QDir(QCoreApplication::applicationDirPath());
    if (dir.dirName() == QLatin1String("build"))
        dir.cdUp(); // gui/
    if (dir.dirName() == QLatin1String("gui"))
        dir.cdUp();
    return dir.absolutePath();
}

QString MainWindow::uvProgram() const
{
    const QString custom = m_uvPath->text().trimmed();
    return custom.isEmpty() ? QStringLiteral("uv") : custom;
}

QStringList MainWindow::buildArgs(QString *error) const
{
    QStringList args;
    args << QStringLiteral("run") << QStringLiteral("python")
         << QStringLiteral("project/follow_object.py");

    const Scenario scenario = currentScenario();
    const Target target = currentTarget();
    const CalibMode calib = currentCalibMode();

    if (scenario == Scenario::Detection) {
        args << QStringLiteral("--no-robot");
    } else if (scenario == Scenario::LabRobot) {
        const QString ip = m_ip->text().trimmed();
        if (ip.isEmpty()) {
            if (error)
                *error = tr("IP robot manquante.");
            return {};
        }
        args << QStringLiteral("--ip") << ip;
        if (currentPoseMode() == PoseMode::Home)
            args << QStringLiteral("--home");
        else
            args << QStringLiteral("--start-here");

        switch (calib) {
        case CalibMode::Load:
            args << QStringLiteral("--load-calib");
            break;
        case CalibMode::Calibrate:
            args << QStringLiteral("--calibrate");
            break;
        case CalibMode::CalibrateScene:
            args << QStringLiteral("--calibrate-scene");
            break;
        case CalibMode::None:
            break;
        }
    } else { // URSim
        if (currentUrsimMode() == UrsimMode::WebcamTable)
            args << QStringLiteral("--webcam-table");
        else
            args << QStringLiteral("--sim");
    }

    const QString camera = m_camera->text().trimmed();
    if (!camera.isEmpty() && camera != QLatin1String("0"))
        args << QStringLiteral("--camera") << camera;
    else if (!camera.isEmpty())
        args << QStringLiteral("--camera") << camera;

    args << QStringLiteral("--color") << m_color->currentText();

    if (target == Target::Shape) {
        const QString path = m_shapePath->text().trimmed();
        if (path.isEmpty())
            args << QStringLiteral("--shape");
        else
            args << QStringLiteral("--shape") << path;
    }

    if (m_gain->value() != 1.0)
        args << QStringLiteral("--gain") << QString::number(m_gain->value());
    if (m_feedforward->value() != 0.3)
        args << QStringLiteral("--feedforward") << QString::number(m_feedforward->value());
    if (m_maxSpeed->value() != 0.02)
        args << QStringLiteral("--max-speed") << QString::number(m_maxSpeed->value());

    if (scenario != Scenario::Detection) {
        if (m_radius->value() != 0.25)
            args << QStringLiteral("--radius") << QString::number(m_radius->value());
        if (m_followDepth->isChecked()) {
            if (scenario == Scenario::URSim) {
                if (error)
                    *error = tr("--follow-depth nécessite --home ou --start-here (scénario robot lab).");
                return {};
            }
            args << QStringLiteral("--follow-depth");
            if (m_targetDistance->value() != 0.5)
                args << QStringLiteral("--target-distance") << QString::number(m_targetDistance->value());
            if (m_depthGain->value() != 1.0)
                args << QStringLiteral("--depth-gain") << QString::number(m_depthGain->value());
        }
    }

    // Advanced (always emit non-defaults so the preview matches what we run)
    if (m_height->value() != 0.30)
        args << QStringLiteral("--height") << QString::number(m_height->value());
    if (m_minZ->value() != 0.10)
        args << QStringLiteral("--min-z") << QString::number(m_minZ->value());
    if (m_calibStep->value() != 0.05)
        args << QStringLiteral("--calib-step") << QString::number(m_calibStep->value());
    if (m_objectDistance->value() != 0.5)
        args << QStringLiteral("--object-distance") << QString::number(m_objectDistance->value());

    const QString homeFile = m_homeFile->text().trimmed();
    if (!homeFile.isEmpty() && homeFile != QLatin1String("data/follow_home.json"))
        args << QStringLiteral("--home-file") << homeFile;
    const QString calibFile = m_calibFile->text().trimmed();
    if (!calibFile.isEmpty() && calibFile != QLatin1String("data/follow_calibration.json"))
        args << QStringLiteral("--calib-file") << calibFile;

    if (m_duration->value() > 0.0)
        args << QStringLiteral("--duration") << QString::number(m_duration->value());
    if (m_headless->isChecked())
        args << QStringLiteral("--headless");

    if (m_simMountEnable->isChecked()) {
        if (scenario != Scenario::URSim || currentUrsimMode() != UrsimMode::Sim) {
            if (error)
                *error = tr("--sim-mount n'est utile qu'avec le scénario URSim / --sim.");
            return {};
        }
        args << QStringLiteral("--sim-mount")
             << QString::number(m_simMountYaw->value())
             << QString::number(m_simMountFlip->currentData().toInt());
    }

    return args;
}

QString MainWindow::commandPreviewText() const
{
    QString err;
    const QStringList args = buildArgs(&err);
    if (!err.isEmpty() && args.isEmpty())
        return tr("Erreur: %1").arg(err);

    QStringList parts;
    parts << quoteArg(uvProgram());
    for (const QString &a : args)
        parts << quoteArg(a);
    return parts.join(QLatin1Char(' '));
}

void MainWindow::onScenarioChanged()
{
    const Scenario s = currentScenario();
    m_labBox->setVisible(s == Scenario::LabRobot);
    m_ursimBox->setVisible(s == Scenario::URSim);

    const bool robotish = (s == Scenario::LabRobot);
    m_radius->setEnabled(robotish || s == Scenario::URSim);
    m_followDepth->setEnabled(robotish);
    if (!robotish)
        m_followDepth->setChecked(false);
    onFollowDepthToggled(m_followDepth->isChecked());

    if (s == Scenario::LabRobot && m_camera->text().trimmed() == QLatin1String("0"))
        m_camera->setText(QStringLiteral("realsense"));
    if (s == Scenario::Detection && m_camera->text().trimmed() == QLatin1String("realsense"))
        m_camera->setText(QStringLiteral("0"));
}

void MainWindow::onTargetChanged()
{
    const bool shape = (currentTarget() == Target::Shape);
    m_shapePath->setEnabled(shape);
    m_browseShape->setEnabled(shape);
}

void MainWindow::onFollowDepthToggled(bool on)
{
    m_targetDistance->setEnabled(on);
    m_depthGain->setEnabled(on);
    if (m_commandPreview)
        refreshCommandPreview();
}

void MainWindow::onBrowseShape()
{
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Photo de référence"), repoRoot() + QStringLiteral("/data"),
        tr("Images (*.jpg *.jpeg *.png);;Tous (*)"));
    if (!path.isEmpty())
        m_shapePath->setText(path);
}

void MainWindow::onBrowseHomeFile()
{
    const QString path = QFileDialog::getSaveFileName(
        this, tr("Fichier home"), repoRoot() + QStringLiteral("/data/follow_home.json"),
        tr("JSON (*.json);;Tous (*)"));
    if (!path.isEmpty())
        m_homeFile->setText(path);
}

void MainWindow::onBrowseCalibFile()
{
    const QString path = QFileDialog::getSaveFileName(
        this, tr("Fichier calibration"),
        repoRoot() + QStringLiteral("/data/follow_calibration.json"),
        tr("JSON (*.json);;Tous (*)"));
    if (!path.isEmpty())
        m_calibFile->setText(path);
}

void MainWindow::onBrowseRepoRoot()
{
    const QString path = QFileDialog::getExistingDirectory(this, tr("Racine du dépôt"), repoRoot());
    if (!path.isEmpty())
        m_repoRoot->setText(path);
}

void MainWindow::onBrowseUv()
{
    const QString path = QFileDialog::getOpenFileName(this, tr("Binaire uv"), QString(),
                                                      tr("Tous (*)"));
    if (!path.isEmpty())
        m_uvPath->setText(path);
}

void MainWindow::refreshCommandPreview()
{
    if (!m_commandPreview)
        return;
    m_commandPreview->setText(commandPreviewText());
}

void MainWindow::setRunning(bool running)
{
    m_launch->setEnabled(!running);
    m_stop->setEnabled(running);
}

void MainWindow::appendLog(const QString &text)
{
    m_logs->moveCursor(QTextCursor::End);
    m_logs->insertPlainText(text);
    m_logs->moveCursor(QTextCursor::End);
}

void MainWindow::stopProcess(bool forceKill)
{
    m_killTimer->stop();
    if (m_process->state() == QProcess::NotRunning)
        return;
    if (forceKill) {
        m_process->kill();
        m_process->waitForFinished(2000);
        return;
    }
    appendLog(tr("\n[launcher] terminate…\n"));
    m_process->terminate();
    m_killTimer->start(2000);
}

void MainWindow::onLaunch()
{
    if (m_process->state() != QProcess::NotRunning)
        return;

    QString err;
    const QStringList args = buildArgs(&err);
    if (args.isEmpty()) {
        QMessageBox::warning(this, tr("Configuration invalide"), err);
        return;
    }

    const QString root = repoRoot();
    if (!QFileInfo(QDir(root).filePath(QStringLiteral("project/follow_object.py"))).isFile()) {
        QMessageBox::warning(
            this, tr("Dépôt introuvable"),
            tr("Impossible de trouver project/follow_object.py sous:\n%1\n"
               "Indiquez la racine du dépôt dans Avancé.")
                .arg(root));
        return;
    }

    m_process->setWorkingDirectory(root);
    m_process->setProgram(uvProgram());
    m_process->setArguments(args);

    appendLog(tr("\n[launcher] cwd=%1\n[launcher] %2\n")
                  .arg(root, commandPreviewText()));
    m_process->start();
    if (!m_process->waitForStarted(5000)) {
        QMessageBox::critical(this, tr("Échec du lancement"),
                              tr("Impossible de démarrer « %1 ».\n%2")
                                  .arg(uvProgram(), m_process->errorString()));
        setRunning(false);
        return;
    }
    setRunning(true);
}

void MainWindow::onStop()
{
    stopProcess(false);
}

void MainWindow::onClearLogs()
{
    m_logs->clear();
}

void MainWindow::onProcessReadyRead()
{
    appendLog(QString::fromUtf8(m_process->readAll()));
}

void MainWindow::onProcessFinished(int exitCode, QProcess::ExitStatus status)
{
    m_killTimer->stop();
    setRunning(false);
    if (status == QProcess::CrashExit)
        appendLog(tr("\n[launcher] process crashed (code %1)\n").arg(exitCode));
    else
        appendLog(tr("\n[launcher] exit %1\n").arg(exitCode));
}

void MainWindow::onProcessError(QProcess::ProcessError error)
{
    if (error == QProcess::FailedToStart) {
        setRunning(false);
        appendLog(tr("\n[launcher] failed to start: %1\n").arg(m_process->errorString()));
    }
}

void MainWindow::onKillTimeout()
{
    if (m_process->state() != QProcess::NotRunning) {
        appendLog(tr("[launcher] kill\n"));
        m_process->kill();
    }
}
