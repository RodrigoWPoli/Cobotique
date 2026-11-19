#include "MainWindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("follow_launcher"));
    QApplication::setOrganizationName(QStringLiteral("Cobotique"));

    MainWindow window;
    window.show();
    return app.exec();
}
