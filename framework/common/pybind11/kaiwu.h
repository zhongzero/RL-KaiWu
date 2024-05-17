
#ifndef KAIWU_H_
#define KAIWU_H_

#include <string>
namespace kaiwu
{
    class KaiWu
    {
        public:
        KaiWu(const std::string &name);

        void setName(const std::string &name);

        const std::string &getName() const { return m_name; };

        private:
        std::string m_name;

    };
}

#endif
