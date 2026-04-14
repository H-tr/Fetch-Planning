#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct FetchBase
{
    static constexpr char* name = "fetchbase";
    static constexpr std::size_t dimension = 3;
    static constexpr std::size_t n_spheres = 14;
    static constexpr float min_radius = 0.06599999964237213;
    static constexpr float max_radius = 0.23999999463558197;
    static constexpr std::size_t resolution = 32;

    static constexpr std::array<std::string_view, dimension> joint_names = {"base_x_joint", "base_y_joint", "base_theta_joint"};
    static constexpr char* end_effector = "base_link";

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        20.0, 20.0, 6.2831854820251465
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        -10.0, -10.0, -3.1415927410125732
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        0.05000000074505806, 0.05000000074505806, 0.15915493667125702
    };

    static inline void scale_configuration(Configuration& q) noexcept
    {
        q = q * Configuration(s_m) + Configuration(s_a);
    }

    static inline void descale_configuration(Configuration& q) noexcept
    {
        q = (q - Configuration(s_a)) * Configuration(d_m);
    }

    template <std::size_t rake>
    static inline void scale_configuration_block(ConfigurationBlock<rake> &q) noexcept
    {
        q[0] = -10.0 + (q[0] * 20.0);
q[1] = -10.0 + (q[1] * 20.0);
q[2] = -3.1415927410125732 + (q[2] * 6.2831854820251465);

    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        q[0] = 0.05000000074505806 * (q[0] - -10.0);
q[1] = 0.05000000074505806 * (q[1] - -10.0);
q[2] = 0.15915493667125702 * (q[2] - -3.1415927410125732);

    }

    inline static auto space_measure() noexcept -> float
    {
        return 2513.2741928100586;
    }

    template <std::size_t rake>
    static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
    {
        std::array<FloatVector<rake, 1>, 3> v;
        std::array<FloatVector<rake, 1>, 56> y;

           v[0] = cos(x[2]);
   y[0] = -0.12 * v[0] + x[0];
   v[1] = sin(x[2]);
   y[1] = -0.12 * v[1] + x[1];
   y[4] = 0.225 * v[0] + x[0];
   y[5] = 0.225 * v[1] + x[1];
   v[2] = - v[1];
   y[8] = 0.08 * v[0] + -0.06 * v[2] + x[0];
   y[9] = 0.08 * v[1] + -0.06 * v[0] + x[1];
   y[12] = 0.215 * v[0] + -0.07 * v[2] + x[0];
   y[13] = 0.215 * v[1] + -0.07 * v[0] + x[1];
   y[16] = 0.185 * v[0] + -0.135 * v[2] + x[0];
   y[17] = 0.185 * v[1] + -0.135 * v[0] + x[1];
   y[20] = 0.13 * v[0] + -0.185 * v[2] + x[0];
   y[21] = 0.13 * v[1] + -0.185 * v[0] + x[1];
   y[24] = 0.065 * v[0] + -0.2 * v[2] + x[0];
   y[25] = 0.065 * v[1] + -0.2 * v[0] + x[1];
   y[28] = 0.01 * v[0] + -0.2 * v[2] + x[0];
   y[29] = 0.01 * v[1] + -0.2 * v[0] + x[1];
   y[32] = 0.08 * v[0] + 0.06 * v[2] + x[0];
   y[33] = 0.08 * v[1] + 0.06 * v[0] + x[1];
   y[36] = 0.215 * v[0] + 0.07 * v[2] + x[0];
   y[37] = 0.215 * v[1] + 0.07 * v[0] + x[1];
   y[40] = 0.185 * v[0] + 0.135 * v[2] + x[0];
   y[41] = 0.185 * v[1] + 0.135 * v[0] + x[1];
   y[44] = 0.13 * v[0] + 0.185 * v[2] + x[0];
   y[45] = 0.13 * v[1] + 0.185 * v[0] + x[1];
   y[48] = 0.065 * v[0] + 0.2 * v[2] + x[0];
   y[49] = 0.065 * v[1] + 0.2 * v[0] + x[1];
   y[52] = 0.01 * v[0] + 0.2 * v[2] + x[0];
   y[53] = 0.01 * v[1] + 0.2 * v[0] + x[1];
   // dependent variables without operations
   y[2] = 0.182;
   y[3] = 0.239999994635582;
   y[6] = 0.31;
   y[7] = 0.0659999996423721;
   y[10] = 0.16;
   y[11] = 0.219999998807907;
   y[14] = 0.31;
   y[15] = 0.0659999996423721;
   y[18] = 0.31;
   y[19] = 0.0659999996423721;
   y[22] = 0.31;
   y[23] = 0.0659999996423721;
   y[26] = 0.31;
   y[27] = 0.0659999996423721;
   y[30] = 0.31;
   y[31] = 0.0659999996423721;
   y[34] = 0.16;
   y[35] = 0.219999998807907;
   y[38] = 0.31;
   y[39] = 0.0659999996423721;
   y[42] = 0.31;
   y[43] = 0.0659999996423721;
   y[46] = 0.31;
   y[47] = 0.0659999996423721;
   y[50] = 0.31;
   y[51] = 0.0659999996423721;
   y[54] = 0.31;
   y[55] = 0.0659999996423721;


        for (auto i = 0U; i < 14; ++i)
        {
            out.x[i] = y[i * 4 + 0];
            out.y[i] = y[i * 4 + 1];
            out.z[i] = y[i * 4 + 2];
            out.r[i] = y[i * 4 + 3];
        }
    }

    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
        std::array<FloatVector<rake, 1>, 3> v;
        std::array<FloatVector<rake, 1>, 60> y;

           v[0] = cos(x[2]);
   y[0] = -0.12 * v[0] + x[0];
   v[1] = sin(x[2]);
   y[1] = -0.12 * v[1] + x[1];
   y[4] = 0.225 * v[0] + x[0];
   y[5] = 0.225 * v[1] + x[1];
   v[2] = - v[1];
   y[8] = 0.08 * v[0] + -0.06 * v[2] + x[0];
   y[9] = 0.08 * v[1] + -0.06 * v[0] + x[1];
   y[12] = 0.215 * v[0] + -0.07 * v[2] + x[0];
   y[13] = 0.215 * v[1] + -0.07 * v[0] + x[1];
   y[16] = 0.185 * v[0] + -0.135 * v[2] + x[0];
   y[17] = 0.185 * v[1] + -0.135 * v[0] + x[1];
   y[20] = 0.13 * v[0] + -0.185 * v[2] + x[0];
   y[21] = 0.13 * v[1] + -0.185 * v[0] + x[1];
   y[24] = 0.065 * v[0] + -0.2 * v[2] + x[0];
   y[25] = 0.065 * v[1] + -0.2 * v[0] + x[1];
   y[28] = 0.01 * v[0] + -0.2 * v[2] + x[0];
   y[29] = 0.01 * v[1] + -0.2 * v[0] + x[1];
   y[32] = 0.08 * v[0] + 0.06 * v[2] + x[0];
   y[33] = 0.08 * v[1] + 0.06 * v[0] + x[1];
   y[36] = 0.215 * v[0] + 0.07 * v[2] + x[0];
   y[37] = 0.215 * v[1] + 0.07 * v[0] + x[1];
   y[40] = 0.185 * v[0] + 0.135 * v[2] + x[0];
   y[41] = 0.185 * v[1] + 0.135 * v[0] + x[1];
   y[44] = 0.13 * v[0] + 0.185 * v[2] + x[0];
   y[45] = 0.13 * v[1] + 0.185 * v[0] + x[1];
   y[48] = 0.065 * v[0] + 0.2 * v[2] + x[0];
   y[49] = 0.065 * v[1] + 0.2 * v[0] + x[1];
   y[52] = 0.01 * v[0] + 0.2 * v[2] + x[0];
   y[53] = 0.01 * v[1] + 0.2 * v[0] + x[1];
   y[56] = -0.0201118849217892 * v[0] + -5.20417042793042e-17 * v[2] + x[0];
   y[57] = -0.0201118849217892 * v[1] + -5.20417042793042e-17 * v[0] + x[1];
   // dependent variables without operations
   y[2] = 0.182;
   y[3] = 0.239999994635582;
   y[6] = 0.31;
   y[7] = 0.0659999996423721;
   y[10] = 0.16;
   y[11] = 0.219999998807907;
   y[14] = 0.31;
   y[15] = 0.0659999996423721;
   y[18] = 0.31;
   y[19] = 0.0659999996423721;
   y[22] = 0.31;
   y[23] = 0.0659999996423721;
   y[26] = 0.31;
   y[27] = 0.0659999996423721;
   y[30] = 0.31;
   y[31] = 0.0659999996423721;
   y[34] = 0.16;
   y[35] = 0.219999998807907;
   y[38] = 0.31;
   y[39] = 0.0659999996423721;
   y[42] = 0.31;
   y[43] = 0.0659999996423721;
   y[46] = 0.31;
   y[47] = 0.0659999996423721;
   y[50] = 0.31;
   y[51] = 0.0659999996423721;
   y[54] = 0.31;
   y[55] = 0.0659999996423721;
   y[58] = 0.188239961862564;
   y[59] = 0.340082824230194;


        Debug output;

        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[0],
                y[1],
                y[2],
                y[3]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[4],
                y[5],
                y[6],
                y[7]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[8],
                y[9],
                y[10],
                y[11]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[12],
                y[13],
                y[14],
                y[15]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[16],
                y[17],
                y[18],
                y[19]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[20],
                y[21],
                y[22],
                y[23]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[24],
                y[25],
                y[26],
                y[27]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[28],
                y[29],
                y[30],
                y[31]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[32],
                y[33],
                y[34],
                y[35]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[36],
                y[37],
                y[38],
                y[39]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[40],
                y[41],
                y[42],
                y[43]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[44],
                y[45],
                y[46],
                y[47]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[48],
                y[49],
                y[50],
                y[51]));
        
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[52],
                y[53],
                y[54],
                y[55]));
        

        

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, 3> v;
        std::array<FloatVector<rake, 1>, 60> y;

           v[0] = cos(x[2]);
   y[0] = -0.12 * v[0] + x[0];
   v[1] = sin(x[2]);
   y[1] = -0.12 * v[1] + x[1];
   y[4] = 0.225 * v[0] + x[0];
   y[5] = 0.225 * v[1] + x[1];
   v[2] = - v[1];
   y[8] = 0.08 * v[0] + -0.06 * v[2] + x[0];
   y[9] = 0.08 * v[1] + -0.06 * v[0] + x[1];
   y[12] = 0.215 * v[0] + -0.07 * v[2] + x[0];
   y[13] = 0.215 * v[1] + -0.07 * v[0] + x[1];
   y[16] = 0.185 * v[0] + -0.135 * v[2] + x[0];
   y[17] = 0.185 * v[1] + -0.135 * v[0] + x[1];
   y[20] = 0.13 * v[0] + -0.185 * v[2] + x[0];
   y[21] = 0.13 * v[1] + -0.185 * v[0] + x[1];
   y[24] = 0.065 * v[0] + -0.2 * v[2] + x[0];
   y[25] = 0.065 * v[1] + -0.2 * v[0] + x[1];
   y[28] = 0.01 * v[0] + -0.2 * v[2] + x[0];
   y[29] = 0.01 * v[1] + -0.2 * v[0] + x[1];
   y[32] = 0.08 * v[0] + 0.06 * v[2] + x[0];
   y[33] = 0.08 * v[1] + 0.06 * v[0] + x[1];
   y[36] = 0.215 * v[0] + 0.07 * v[2] + x[0];
   y[37] = 0.215 * v[1] + 0.07 * v[0] + x[1];
   y[40] = 0.185 * v[0] + 0.135 * v[2] + x[0];
   y[41] = 0.185 * v[1] + 0.135 * v[0] + x[1];
   y[44] = 0.13 * v[0] + 0.185 * v[2] + x[0];
   y[45] = 0.13 * v[1] + 0.185 * v[0] + x[1];
   y[48] = 0.065 * v[0] + 0.2 * v[2] + x[0];
   y[49] = 0.065 * v[1] + 0.2 * v[0] + x[1];
   y[52] = 0.01 * v[0] + 0.2 * v[2] + x[0];
   y[53] = 0.01 * v[1] + 0.2 * v[0] + x[1];
   y[56] = -0.0201118849217892 * v[0] + -5.20417042793042e-17 * v[2] + x[0];
   y[57] = -0.0201118849217892 * v[1] + -5.20417042793042e-17 * v[0] + x[1];
   // dependent variables without operations
   y[2] = 0.182;
   y[3] = 0.239999994635582;
   y[6] = 0.31;
   y[7] = 0.0659999996423721;
   y[10] = 0.16;
   y[11] = 0.219999998807907;
   y[14] = 0.31;
   y[15] = 0.0659999996423721;
   y[18] = 0.31;
   y[19] = 0.0659999996423721;
   y[22] = 0.31;
   y[23] = 0.0659999996423721;
   y[26] = 0.31;
   y[27] = 0.0659999996423721;
   y[30] = 0.31;
   y[31] = 0.0659999996423721;
   y[34] = 0.16;
   y[35] = 0.219999998807907;
   y[38] = 0.31;
   y[39] = 0.0659999996423721;
   y[42] = 0.31;
   y[43] = 0.0659999996423721;
   y[46] = 0.31;
   y[47] = 0.0659999996423721;
   y[50] = 0.31;
   y[51] = 0.0659999996423721;
   y[54] = 0.31;
   y[55] = 0.0659999996423721;
   y[58] = 0.188239961862564;
   y[59] = 0.340082824230194;

        





//
// environment vs. robot collisions
//

// base_link
if (sphere_environment_in_collision(environment,
                                    y[56],
                                    y[57],
                                    y[58],
                                    y[59]))
{
    
    
    if (sphere_environment_in_collision(environment,
                                        y[0],
                                        y[1],
                                        y[2],
                                        y[3]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[4],
                                        y[5],
                                        y[6],
                                        y[7]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[8],
                                        y[9],
                                        y[10],
                                        y[11]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[12],
                                        y[13],
                                        y[14],
                                        y[15]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[16],
                                        y[17],
                                        y[18],
                                        y[19]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[20],
                                        y[21],
                                        y[22],
                                        y[23]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[24],
                                        y[25],
                                        y[26],
                                        y[27]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[28],
                                        y[29],
                                        y[30],
                                        y[31]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[32],
                                        y[33],
                                        y[34],
                                        y[35]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[36],
                                        y[37],
                                        y[38],
                                        y[39]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[40],
                                        y[41],
                                        y[42],
                                        y[43]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[44],
                                        y[45],
                                        y[46],
                                        y[47]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[48],
                                        y[49],
                                        y[50],
                                        y[51]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[52],
                                        y[53],
                                        y[54],
                                        y[55]))
    {
        return false;
    }
    
}



//
// robot self-collisions
//





        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const ConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, 0> v;
        std::array<FloatVector<rake, 1>, 72> y;

           y[63] = cos(x[2]);
   y[0] = -0.12 * y[63] + x[0];
   y[64] = sin(x[2]);
   y[1] = -0.12 * y[64] + x[1];
   y[4] = 0.225 * y[63] + x[0];
   y[5] = 0.225 * y[64] + x[1];
   y[66] = - y[64];
   y[8] = 0.08 * y[63] + -0.06 * y[66] + x[0];
   y[9] = 0.08 * y[64] + -0.06 * y[63] + x[1];
   y[12] = 0.215 * y[63] + -0.07 * y[66] + x[0];
   y[13] = 0.215 * y[64] + -0.07 * y[63] + x[1];
   y[16] = 0.185 * y[63] + -0.135 * y[66] + x[0];
   y[17] = 0.185 * y[64] + -0.135 * y[63] + x[1];
   y[20] = 0.13 * y[63] + -0.185 * y[66] + x[0];
   y[21] = 0.13 * y[64] + -0.185 * y[63] + x[1];
   y[24] = 0.065 * y[63] + -0.2 * y[66] + x[0];
   y[25] = 0.065 * y[64] + -0.2 * y[63] + x[1];
   y[28] = 0.01 * y[63] + -0.2 * y[66] + x[0];
   y[29] = 0.01 * y[64] + -0.2 * y[63] + x[1];
   y[32] = 0.08 * y[63] + 0.06 * y[66] + x[0];
   y[33] = 0.08 * y[64] + 0.06 * y[63] + x[1];
   y[36] = 0.215 * y[63] + 0.07 * y[66] + x[0];
   y[37] = 0.215 * y[64] + 0.07 * y[63] + x[1];
   y[40] = 0.185 * y[63] + 0.135 * y[66] + x[0];
   y[41] = 0.185 * y[64] + 0.135 * y[63] + x[1];
   y[44] = 0.13 * y[63] + 0.185 * y[66] + x[0];
   y[45] = 0.13 * y[64] + 0.185 * y[63] + x[1];
   y[48] = 0.065 * y[63] + 0.2 * y[66] + x[0];
   y[49] = 0.065 * y[64] + 0.2 * y[63] + x[1];
   y[52] = 0.01 * y[63] + 0.2 * y[66] + x[0];
   y[53] = 0.01 * y[64] + 0.2 * y[63] + x[1];
   y[56] = -0.0201118849217892 * y[63] + -5.20417042793042e-17 * y[66] + x[0];
   y[57] = -0.0201118849217892 * y[64] + -5.20417042793042e-17 * y[63] + x[1];
   // variable duplicates: 1
   y[67] = y[63];
   // dependent variables without operations
   y[2] = 0.182;
   y[3] = 0.239999994635582;
   y[6] = 0.31;
   y[7] = 0.0659999996423721;
   y[10] = 0.16;
   y[11] = 0.219999998807907;
   y[14] = 0.31;
   y[15] = 0.0659999996423721;
   y[18] = 0.31;
   y[19] = 0.0659999996423721;
   y[22] = 0.31;
   y[23] = 0.0659999996423721;
   y[26] = 0.31;
   y[27] = 0.0659999996423721;
   y[30] = 0.31;
   y[31] = 0.0659999996423721;
   y[34] = 0.16;
   y[35] = 0.219999998807907;
   y[38] = 0.31;
   y[39] = 0.0659999996423721;
   y[42] = 0.31;
   y[43] = 0.0659999996423721;
   y[46] = 0.31;
   y[47] = 0.0659999996423721;
   y[50] = 0.31;
   y[51] = 0.0659999996423721;
   y[54] = 0.31;
   y[55] = 0.0659999996423721;
   y[58] = 0.188239961862564;
   y[59] = 0.340082824230194;
   y[60] = x[0];
   y[61] = x[1];
   y[62] = 0.;
   y[65] = 0.;
   y[68] = 0.;
   y[69] = 0.;
   y[70] = 0.;
   y[71] = 1.;

        





//
// environment vs. robot collisions
//

// base_link
if (sphere_environment_in_collision(environment,
                                    y[56],
                                    y[57],
                                    y[58],
                                    y[59]))
{
    
    
    if (sphere_environment_in_collision(environment,
                                        y[0],
                                        y[1],
                                        y[2],
                                        y[3]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[4],
                                        y[5],
                                        y[6],
                                        y[7]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[8],
                                        y[9],
                                        y[10],
                                        y[11]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[12],
                                        y[13],
                                        y[14],
                                        y[15]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[16],
                                        y[17],
                                        y[18],
                                        y[19]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[20],
                                        y[21],
                                        y[22],
                                        y[23]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[24],
                                        y[25],
                                        y[26],
                                        y[27]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[28],
                                        y[29],
                                        y[30],
                                        y[31]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[32],
                                        y[33],
                                        y[34],
                                        y[35]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[36],
                                        y[37],
                                        y[38],
                                        y[39]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[40],
                                        y[41],
                                        y[42],
                                        y[43]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[44],
                                        y[45],
                                        y[46],
                                        y[47]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[48],
                                        y[49],
                                        y[50],
                                        y[51]))
    {
        return false;
    }
    
    
    if (sphere_environment_in_collision(environment,
                                        y[52],
                                        y[53],
                                        y[54],
                                        y[55]))
    {
        return false;
    }
    
}



//
// robot self-collisions
//





        // attaching at base_link
        set_attachment_pose(environment, to_isometry(&y[60]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment))
        {
            return false;
        }

        //
        // attachment vs. robot collisions
        //

        

        return true;
    }

    static inline auto eefk(const std::array<float, 3> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, 0> v;
        std::array<float, 12> y;

           y[3] = cos(x[2]);
   y[4] = sin(x[2]);
   y[6] = - y[4];
   // variable duplicates: 1
   y[7] = y[3];
   // dependent variables without operations
   y[0] = x[0];
   y[1] = x[1];
   y[2] = 0.;
   y[5] = 0.;
   y[8] = 0.;
   y[9] = 0.;
   y[10] = 0.;
   y[11] = 1.;


        return to_isometry(y.data());
    }
};
}

// NOLINTEND(*-magic-numbers)
