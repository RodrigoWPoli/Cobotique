#pragma once

#include <QMainWindow>
#include <QProcess>
#include <QStringList>
#include <QTimer>

class QButtonGroup;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QRadioButton;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onScenarioChanged();
    void onTargetChanged();
    void onFollowDepthToggled(bool on);
    void onBrowseShape();
    void onBrowseHomeFile();
    void onBrowseCalibFile();
    void onBrowseRepoRoot();
    void onBrowseUv();
    void refreshCommandPreview();
    void onLaunch();
    void onStop();
    void onClearLogs();
    void onProcessReadyRead();
    void onProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onProcessError(QProcess::ProcessError error);
    void onKillTimeout();

private:
    enum class Scenario { Detection, LabRobot, URSim };
    enum class Target { ColorOnly, Shape };
    enum class PoseMode { Home, StartHere };
    enum class CalibMode { Load, Calibrate, CalibrateScene, None };
    enum class UrsimMode { Sim, WebcamTable };

    Scenario currentScenario() const;
    Target currentTarget() const;
    PoseMode currentPoseMode() const;
    CalibMode currentCalibMode() const;
    UrsimMode currentUrsimMode() const;

    QString repoRoot() const;
    QString uvProgram() const;
    QStringList buildArgs(QString *error) const;
    QString commandPreviewText() const;
    void setRunning(bool running);
    void appendLog(const QString &text);
    void stopProcess(bool forceKill);

    void buildUi();
    QWidget *buildScenarioBox();
    QWidget *buildTargetBox();
    QWidget *buildUsefulBox();
    QWidget *buildLabBox();
    QWidget *buildUrsimBox();
    QWidget *buildAdvancedBox();
    QWidget *buildRunBox();

    // Scenario
    QRadioButton *m_scDetection = nullptr;
    QRadioButton *m_scLab = nullptr;
    QRadioButton *m_scUrsim = nullptr;
    QButtonGroup *m_scenarioGroup = nullptr;

    // Target
    QRadioButton *m_tgColor = nullptr;
    QRadioButton *m_tgShape = nullptr;
    QButtonGroup *m_targetGroup = nullptr;
    QComboBox *m_color = nullptr;
    QLineEdit *m_shapePath = nullptr;
    QPushButton *m_browseShape = nullptr;

    // Useful
    QLineEdit *m_camera = nullptr;
    QDoubleSpinBox *m_gain = nullptr;
    QDoubleSpinBox *m_feedforward = nullptr;
    QDoubleSpinBox *m_maxSpeed = nullptr;
    QDoubleSpinBox *m_radius = nullptr;
    QCheckBox *m_followDepth = nullptr;
    QDoubleSpinBox *m_targetDistance = nullptr;
    QDoubleSpinBox *m_depthGain = nullptr;
    QGroupBox *m_usefulBox = nullptr;

    // Lab
    QGroupBox *m_labBox = nullptr;
    QLineEdit *m_ip = nullptr;
    QRadioButton *m_poseHome = nullptr;
    QRadioButton *m_poseStartHere = nullptr;
    QButtonGroup *m_poseGroup = nullptr;
    QComboBox *m_calibMode = nullptr;

    // URSim
    QGroupBox *m_ursimBox = nullptr;
    QRadioButton *m_ursimSim = nullptr;
    QRadioButton *m_ursimWebcam = nullptr;
    QButtonGroup *m_ursimGroup = nullptr;

    // Advanced
    QGroupBox *m_advancedBox = nullptr;
    QDoubleSpinBox *m_height = nullptr;
    QDoubleSpinBox *m_minZ = nullptr;
    QDoubleSpinBox *m_calibStep = nullptr;
    QDoubleSpinBox *m_objectDistance = nullptr;
    QLineEdit *m_homeFile = nullptr;
    QLineEdit *m_calibFile = nullptr;
    QDoubleSpinBox *m_duration = nullptr;
    QCheckBox *m_headless = nullptr;
    QCheckBox *m_simMountEnable = nullptr;
    QDoubleSpinBox *m_simMountYaw = nullptr;
    QComboBox *m_simMountFlip = nullptr;
    QLineEdit *m_repoRoot = nullptr;
    QLineEdit *m_uvPath = nullptr;

    // Run
    QLineEdit *m_commandPreview = nullptr;
    QPushButton *m_launch = nullptr;
    QPushButton *m_stop = nullptr;
    QPushButton *m_clearLogs = nullptr;
    QPlainTextEdit *m_logs = nullptr;

    QProcess *m_process = nullptr;
    QTimer *m_killTimer = nullptr;
};
